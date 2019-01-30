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
"""Defines Params base class, used for defining class/function parameters."""

from __future__ import absolute_import
from __future__ import print_function

import ast
import copy
import inspect
import re
import sys

import six
import tensorflow as tf


def _QuoteString(s):
  """Quotes a string with appropriate quotes and escaping.

  This performs lite escaping by choosing enclosing quotation marks that would
  escape the least (either single or double quotes) and escaping those quotes
  and the backslash. Note that this does not escape newlines. If the string
  contains embedded newlines, they will be output verbatim.

  Args:
    s: String to quote.
  Returns:
    Quotes string (possibly multiline).
  """
  single_quote_count = s.count('\'')
  double_quote_count = s.count('"')
  quote_delim = '\'' if single_quote_count <= double_quote_count else '"'
  # Apply escaping to the chosen quote character and the backslash.
  encoded = re.sub(r'([%s\\])' % (quote_delim,), r'\\\1', s)
  return quote_delim + encoded + quote_delim


def _UnquoteString(quoted):
  if quoted and quoted[0] in ['"', '\'']:
    # Note that only the limited set of escaping produced by _QuoteString is
    # supported.
    contents = quoted.strip(quoted[0])
    return re.sub(r"""\\([\\'"])""", r'\1', contents)
  else:
    # Just return literal text.
    return quoted


def _EndsWithTerminalQuote(s, quote_char):
  """Returns whether a string ends with a valid terminal quote."""
  endm = re.search(r'(\\*)%s$' % quote_char, s)
  if not endm:
    return False
  backslashes = endm.group(1)
  if len(backslashes) % 2 == 0:
    # Even number of backslashes preceding the quote means the quote is
    # not escaped.
    return True
  else:
    # Terminal quote is escaped.
    return False


class _SortedDict(dict):
  """A dict with a __repr__ that is always sorted by key."""

  def __repr__(self):
    return '{' + ', '.join(
        '%r: %r' % item for item in sorted(self.items())) + '}'


class _Param(object):
  """Stores data for a single parameter."""

  def __init__(self, name, default_value, description):
    self._name = name
    self._value = default_value
    self._description = description

  def __eq__(self, other):
    # pylint: disable=protected-access
    return self._name == other._name and self._value == other._value

  # Deep copy the value only if it is supported.
  def __deepcopy__(self, memo):
    try:
      value = copy.deepcopy(self._value, memo)
    except:  # pylint: disable=bare-except
      if self._value.__class__.__name__ == 'Tensor':
        # In case self._value is a tensor, let's just make a reference.
        # Q(yonghui): Is there a better / more reliable way of detecting the
        # type of self._value without importing more modules?
        value = self._value
      else:
        raise
    p = _Param(self._name, value, self._description)
    # Q(yonghui): Is this the right use of memo.
    memo[id(self)] = p
    return p

  def ToString(self, nested_depth):
    """Prints the parameter as a string."""

    def GetRepr(val):
      """Get the representation of `val`."""
      if isinstance(val, Params):
        return _SortedDict({k: GetRepr(v) for k, v in val.IterParams()})
      if isinstance(val, dict):
        return _SortedDict({k: GetRepr(v) for k, v in six.iteritems(val)})
      if isinstance(val, (list, tuple)):
        return type(val)([GetRepr(v) for v in val])
      # NOTE(markmurphy): I introduced Repr() because it's impossible (afaik) to
      # overwrite the __str__ or __repr__ method of a types.FunctionType object.
      if hasattr(val, 'Repr'):
        return val.Repr()
      return val

    nested_indent = '  ' * nested_depth
    if isinstance(self._value, Params):
      # pylint: disable=protected-access
      value_str = self._value._ToString(nested_depth)
    elif isinstance(self._value, six.string_types):
      return '"%s"' % self._value
    else:
      value_str = str(GetRepr(self._value))
    return '%s%s: %s' % (nested_indent, self._name, value_str)

  def Set(self, value):
    # Note that we don't make a copy of Params objects.
    # TODO(sadovsky): Maybe add safeguard to ensure that Params object is not
    # owned by other Params objects.
    self._value = value

  def Get(self):
    return self._value


class Params(object):
  """Stores data for a set of parameters.

  Provides attribute-based API, e.g. "params.foo = 5".
  Uses internal {'name': _Param} dict for storing parameter data.
  """

  def __init__(self):
    self.__dict__['_immutable'] = False
    self._params = {}  # name => _Param

  def __setattr__(self, name, value):
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    if name == '_params' or name == '_immutable':
      self.__dict__[name] = value
    else:
      try:
        self._params[name].Set(value)
      except KeyError:
        raise AttributeError(name)

  def __getattr__(self, name):
    if name == '_params' or name == '_immutable':
      return self.__dict__[name]
    try:
      return self._params[name].Get()
    except KeyError:
      # cPickle expects __getattr__ to raise AttributeError, not KeyError.
      raise AttributeError(name)

  def __dir__(self):
    return sorted(self._params.keys())

  def __len__(self):
    return len(self._params)

  # Note: This gets called by _Param.__eq__() on nested Params objects.
  def __eq__(self, other):
    return self._params == other._params  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __str__(self):
    return self._ToString(0)

  def _ToString(self, nested_depth):
    # Note: We use iteritems() below so as to sort by name.
    sorted_param_strs = [
        v.ToString(nested_depth + 1)
        for (_, v) in sorted(six.iteritems(self._params))
    ]
    nested_indent = '  ' * nested_depth
    return '{\n%s\n%s}' % ('\n'.join(sorted_param_strs), nested_indent)

  # Override __deepcopy__ so that copy.deepcopy(self._params) properly
  # deep-copies nested Params objects.
  # TODO(sadovsky): Is it okay not to touch memo?
  def __deepcopy__(self, unused_memo):
    return self.Copy()

  def Copy(self):
    res = type(self)()
    # pylint: disable=protected-access
    res._params = copy.deepcopy(self._params)
    res._immutable = self._immutable
    return res

  # TODO(sadovsky):
  # - Maybe let users specify whether this parameter is allowed to have
  #   value=None, and if not, assert on Get(), like required proto field.
  # - Maybe enforce that value is one of
  #     {number, string, bool, list, dict, Params}.
  def Define(self, name, default_value, description):
    """Defines a parameter.

    Args:
      name: The parameter name. Must only contain lowercase letters, numbers,
          and underscores. Must start with lowercase letter.
      default_value: Default value for this parameter. May be None.
      description: String description of this parameter.

    Raises:
      AttributeError: If parameter 'name' is already defined.
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    assert name is not None and isinstance(
        name,
        six.string_types) and (re.match('^[a-z][a-z0-9_]*$', name) is not None)
    if name in self._params:
      raise AttributeError('Parameter %s is already defined' % name)
    self._params[name] = _Param(name, default_value, description)

  def Freeze(self):
    """Marks this Params as immutable."""
    self._immutable = True

  def _GetNested(self, name):
    """Returns nested param by its name."""
    parts = name.split('.')
    curr = self
    for i, part in enumerate(parts[:-1]):
      # Get the value (nested Params object) associated with name 'part'.
      try:
        is_list = re.match(r'^(.+)\[(.+)\]$', part)
        if is_list:
          part = is_list.group(1)
          list_index = int(is_list.group(2))
        # pylint: disable=protected-access
        curr = curr._params[part].Get()
        if is_list:
          curr = curr[list_index]
      except KeyError:
        raise AttributeError('.'.join(parts[:i + 1]))
      assert isinstance(curr, Params), (
          'Cannot introspect %s for %s' % (type(curr), '.'.join(parts[:i + 1])))
    return curr, parts[-1]

  def Set(self, **kwargs):
    """Sets multiple parameters.

    Dots in names indicate navigation into nested Params objects. We do not
    allow navigation into lists or dicts, and may ban these types altogether in
    favor of string representations.

    Args:
      **kwargs: Name-value pairs to set.

    Returns:
      self
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    for name, value in six.iteritems(kwargs):
      # Get nested param.
      param, key = self._GetNested(name)
      # Update the value associated with key.
      try:
        # pylint: disable=protected-access
        param._params[key].Set(value)
      except KeyError:
        raise AttributeError(name)
    return self

  def Get(self, name):
    """Get parameter.

    Dots in names indicate navigation into nested Params objects. We do not
    allow navigation into lists or dicts, and may ban these types altogether in
    favor of string representations.

    Args:
      name: (str) Name.

    Returns:
      value.

    Raises:
      AttributeError: if parameter is not found
    """
    param, key = self._GetNested(name)
    # Get the value associated with key.
    try:
      # pylint: disable=protected-access
      return param._params[key].Get()
    except KeyError:
      raise AttributeError(name)

  def Delete(self, *args):
    """Deletes multiple parameters.

    Dots in names indicate navigation into nested Params objects. We do not
    allow navigation into lists or dicts, and may ban these types altogether in
    favor of string representations.

    Args:
      *args: List of names.

    Returns:
      self
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    for name in args:
      # Get nested param.
      param, key = self._GetNested(name)
      # Delete the key.
      try:
        # pylint: disable=protected-access
        del param._params[key]
      except KeyError:
        raise AttributeError(name)
    return self

  def IterParams(self):
    """Pythonic dict-like iteration."""
    for name, param in six.iteritems(self._params):
      yield (name, param.Get())

  def ToText(self):
    """Encodes params into a simple text format.

    Each param is represented as a single line in the output.  The param
    name and value is separated by a ":".  The nest param name is
    separated by ".".  For values of non-trivial types (types other than
    int, float, bool, str, and a few, etc.), we just print out the name
    of its type.

    Note that strings are enclosed in appropriate single or double quotes
    (whichever would involve the least escaping) and will have some characters
    backslash escaped. String properties can span multiple lines.

    Returns:
      The encoded text.
    """
    kv = {}

    def GetRepr(val):
      """Get the representation of `val`."""
      if isinstance(val, Params):
        return _SortedDict({k: GetRepr(v) for k, v in val.IterParams()})
      if isinstance(val, dict):
        return _SortedDict({k: GetRepr(v) for k, v in six.iteritems(val)})
      if isinstance(val, (list, tuple)):
        return type(val)([GetRepr(v) for v in val])
      if isinstance(
          val,
          (six.integer_types, float, bool, six.string_types, six.text_type)):
        return val
      if isinstance(val, tf.DType):
        return val.name
      if isinstance(val, type):
        return 'type/' + inspect.getmodule(val).__name__ + '/' + val.__name__
      return type(val).__name__

    def Traverse(p, prefix, kv):
      """Traverses 'p' and inserts key-value pairs to 'kv'."""
      if isinstance(p, Params):
        for key, val in p.IterParams():
          Traverse(val, prefix + '.' + key, kv)
      elif (isinstance(p, (list, tuple)) and
            all(isinstance(x, Params) for x in p)):
        for i, val in enumerate(p):
          Traverse(val, '%s[%d]' % (prefix, i), kv)
      elif isinstance(p, (six.string_types, six.text_type)):
        kv[prefix] = _QuoteString(p)
      else:
        kv[prefix] = str(GetRepr(p))

    Traverse(self, '', kv)
    ret = ''
    for (k, v) in sorted(kv.items()):
      ret += k[1:] + ' : ' + v + '\n'
    return ret

  def FromText(self, text):
    """Merges params specified in 'text' into 'params'.

    'text' follows the simple text format as produced by
    ParamsToSimpleText.  For a param specified in both 'params' and
    'text', overwrites the value in 'params' according to 'text'.
    Params specified in 'text' but not in 'params' are ignored.

    Args:
      text: A text representation of params.
    Raises:
      AttributeError: text contains invalid parameter key
      ValueError: text contains invalid parameter value
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    kv = {}
    string_continue = None  # None or (key, quote, value)
    for line in text.split('\n'):
      # Continuing a multi-line string.
      if string_continue:
        value_stripped = line.rstrip()
        if not _EndsWithTerminalQuote(value_stripped, string_continue[1]):
          # String continues
          string_continue = (string_continue[0], string_continue[1],
                             string_continue[2] + '\n' + line)
          continue
        # String terminates.
        kv[string_continue[0]] = string_continue[2] + '\n' + value_stripped
        string_continue = None
        continue

      # Regular line.
      line = line.strip()
      if not line or line[0] == '#':
        # empty line or comment
        continue
      pair = line.split(':', 1)
      if len(pair) == 2:
        key = pair[0].strip()
        value = pair[1].lstrip()
        value_stripped = value.rstrip()
        # Detect single vs multi-line string start.
        if value and value[0] in ['"', '\'']:
          quote_char = value[0]
          if not _EndsWithTerminalQuote(value[1:], quote_char):
            # Multi-line string.
            string_continue = (key, quote_char, value)
            continue
        kv[key] = value_stripped
    for key, val in six.iteritems(kv):
      old_val = self.Get(key)
      # Converts val (a string) to a best-guessed typed value.
      if isinstance(old_val, bool):
        val = (val and (val != 'False') and (val != 'false'))
      elif isinstance(old_val, int):
        val = int(val)
      elif isinstance(old_val, float):
        val = float(val)
      elif isinstance(old_val, tf.DType):
        val = tf.as_dtype(val)
      elif isinstance(old_val, (six.string_types, six.text_type)):
        val = _UnquoteString(val)
      elif isinstance(old_val, (list, tuple)):
        val = ast.literal_eval(val)
      elif isinstance(old_val, dict):
        val = ast.literal_eval(val) if val != 'dict' else {}
      elif isinstance(old_val, type) or old_val is None:
        if val == 'NoneType':
          val = None
        elif old_val is None and val in ('False', 'false'):
          val = False
        elif old_val is None and val in ('True', 'true'):
          val = True
        else:
          try:
            _, pkg, cls = val.split('/')
            val = getattr(sys.modules[pkg], cls)
          except ValueError as e:
            raise ValueError('Error processing %r : %r with %r' % (key, val, e))
      else:
        raise ValueError('Failed to read a parameter: %r : %r' % (key, val))
      self.Set(**{key: val})
