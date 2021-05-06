# Lint as: python3
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
"""NestedMap dict structure."""

import re
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple,
                    TypeVar, Union)
import lingvo.compat as tf

_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_SQUARE_BRACKET_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]')


NestedMapT = TypeVar('NestedMapT', bound='NestedMap')


class NestedMap(Dict[str, Any]):
  """A simple helper to maintain a dict.

  It is a sub-class of dict with the following extensions/restrictions:
    - It supports attr access to its members (see examples below).
    - Member keys have to be valid identifiers.

  E.g.::

      >>> foo = NestedMap()
      >>> foo['x'] = 10
      >>> foo.y = 20
      >>> assert foo.x * 2 == foo.y
  """

  # Disable pytype attribute checking.
  _HAS_DYNAMIC_ATTRIBUTES = True
  # keys in this list are not allowed in a NestedMap.
  _RESERVED_KEYS = frozenset(dir(dict))
  # sentinel value for deleting keys used in Filter.
  _DELETE = object()

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    for key in self.keys():
      assert isinstance(key, str), (
          'Key in a NestedMap has to be a six.string_types. Currently type: %s,'
          ' value: %s' % (str(type(key)), str(key)))
      NestedMap.CheckKey(key)
      assert key not in NestedMap._RESERVED_KEYS, ('%s is a reserved key' % key)

  def __setitem__(self, key: str, value: Any) -> None:
    # Make sure key is a valid expression and is not one of the reserved
    # attributes.
    assert isinstance(
        key,
        str), ('Key in a NestedMap has to be a string type. Current type: %s, '
               'value: %s' % (str(type(key)), str(key)))
    NestedMap.CheckKey(key)
    assert key not in NestedMap._RESERVED_KEYS, ('%s is a reserved key' % key)
    super().__setitem__(key, value)

  def __setattr__(self, name: str, value: Any) -> None:
    self.__setitem__(name, value)

  def __getattr__(self, name: str) -> Any:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError('%s; available attributes: %s' %
                           (e, sorted(list(self.keys()))))

  def __delattr__(self, name: str) -> None:
    try:
      del self[name]
    except KeyError as e:
      raise AttributeError('%s; available attributes: %s' %
                           (e, sorted(list(self.keys()))))

  def copy(self: NestedMapT) -> NestedMapT:  # pylint: disable=invalid-name
    # Don't delegate w/ super: dict.copy() -> dict.
    return type(self)(self)

  def __deepcopy__(self: NestedMapT, unused_memo) -> NestedMapT:
    """Deep-copies the structure but not the leaf objects."""
    return self.DeepCopy()

  def DeepCopy(self: NestedMapT) -> NestedMapT:
    """Deep-copies the structure but not the leaf objects."""
    return self.Pack(self.Flatten())

  @staticmethod
  def FromNestedDict(
      x: Union[NestedMapT, Dict[str, Any], List[Any], Tuple[Any, ...]]
  ) -> NestedMapT:
    """Converts every dict in nested structure 'x' to a NestedMap."""
    if isinstance(x, dict):
      res = NestedMap()
      for k, v in x.items():
        res[k] = NestedMap.FromNestedDict(v)
      return res
    elif isinstance(x, (list, tuple)):
      return type(x)(NestedMap.FromNestedDict(v) for v in x)
    else:
      return x

  @staticmethod
  def CheckKey(key: str) -> None:
    """Asserts that key is valid NestedMap key."""
    if not (isinstance(key, str) and _NAME_PATTERN.match(key)):
      raise ValueError('Invalid NestedMap key \'{}\''.format(key))

  @staticmethod
  def SquareBracketIndex(key: str) -> Tuple[str, Optional[int]]:
    """Extracts the name and the index from the indexed key (e.g., k[0])."""
    m = _SQUARE_BRACKET_PATTERN.fullmatch(key)
    if not m:
      return key, None
    else:
      return m.groups()[0], int(m.groups()[1])

  def GetItem(self, key: str) -> Any:
    """Gets the value for the nested `key`.

    Names with underscores will be considered as one key.

    Args:
      key: str of the form
        `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.

    Returns:
      The value for the given nested key.

    Raises:
      KeyError: if a key is not present.
      IndexError: when an intermediate item is a list and we try to access
        an element which is out of range.
      TypeError: when an intermediate item is a list and we try to access
        an element of it with a string.
    """
    current = self
    for k in key.split('.'):
      k, idx = self.SquareBracketIndex(k)
      current = current[k]
      if idx is not None:
        current = current[idx]
    return current

  def Get(self, key: str, default: Optional[Any] = None) -> Any:
    """Gets the value for nested `key`, returns `default` if key does not exist.

    Names with underscores will be considered as one key.

    Args:
      key: str of the form
        `([A-Za-z_][A-Za-z0-9_]*)(.[A-Za-z_][A-Za-z0-9_]*)*.`.
      default: Optional default value, defaults to None.

    Returns:
      The value for the given nested key or `default` if the key does not exist.
    """
    try:
      return self.GetItem(key)
    except (KeyError, IndexError, TypeError):
      return default

  def Set(self, key: str, value: Any) -> None:
    r"""Sets the value for a nested key.

    There is limited support for indexing lists when square bracket indexing is
    used, e.g., key[0], key[1], etc. Names with underscores will be considered
    as one key. When key[idx] is set, all of the values with indices before idx
    must be already set. E.g., setting key='a[2]' to value=42 when
    key='a' wasn't referenced before will throw a ValueError. Setting key='a[0]'
    will not.

    Args:
      key: str of the form key_part1.key_part2...key_partN where each key_part
        is of the form `[A-Za-z_][A-Za-z0-9_]*` or
        `[A-Za-z_][A-Za-z0-9_]*\[\d+\]`
      value: The value to insert.

    Raises:
      ValueError if a sub key is not a NestedMap or dict or idx > list length
      for key='key[idx]'.
    """
    current = self
    sub_keys = key.split('.')
    for i, k in enumerate(sub_keys):
      self.CheckKey(k)  # CheckKey allows k to be of form k[\d+]
      k, idx = self.SquareBracketIndex(k)
      if idx is not None:  # this is key with index pointing to a list item.
        # create a list if not there yet.
        if k not in current:
          current[k] = []
        if idx > len(current[k]):
          raise ValueError('Error while setting key {}. The value under {} is a'
                           ' list and the index {} is greater than the len={} '
                           'of this list'.format(key, k, idx, len(current[k])))
        elif idx == len(current[k]):
          current[k].extend([None])  # this None will be overwritten right away.

      # We have reached the terminal node, set the value.
      if i == (len(sub_keys) - 1):
        if idx is None:
          current[k] = value
        else:
          current[k][idx] = value
      else:
        if idx is None:
          if k not in current:
            current[k] = NestedMap()
          current = current[k]
        else:
          if current[k][idx] is None:
            current[k][idx] = NestedMap()
          current = current[k][idx]
        if not isinstance(current, (dict, NestedMap)):
          raise ValueError('Error while setting key {}. Sub key "{}" is of type'
                           ' {} but must be a dict or NestedMap.'
                           ''.format(key, k, type(current)))

  def _RecursiveMap(self: NestedMapT,
                    fn: Callable[[str, Any], Any],
                    flatten: bool = False) -> Union[List[Any], NestedMapT]:
    """Traverse recursively into lists, dicts, and NestedMaps applying `fn`.

    Args:
      fn: The function to apply to each item (leaf node).
      flatten: If true, the result should be a single flat list. Otherwise the
        result will have the same structure as this NestedMap.

    Returns:
      The result of applying fn.
    """

    def Recurse(v: Any, key: str = ''):
      """Helper function for _RecursiveMap."""
      if isinstance(v, dict):
        ret = [] if flatten else type(v)()
        deleted = False
        for k in sorted(v.keys()):
          res = Recurse(v[k], key + '.' + k if key else k)
          if res is self._DELETE:
            deleted = True
            continue
          elif flatten:
            ret += res
          else:
            ret[k] = res
        if not ret and deleted:
          return self._DELETE
        return ret
      elif isinstance(v, list):
        ret = []
        deleted = False
        for i, x in enumerate(v):
          res = Recurse(x, '%s[%d]' % (key, i))
          if res is self._DELETE:
            deleted = True
            continue
          elif flatten:
            ret += res
          else:
            ret.append(res)
        if not ret and deleted:
          return self._DELETE
        return ret
      else:
        ret = fn(key, v)
        if flatten:
          ret = [ret]
        return ret

    res = Recurse(self)
    if res is self._DELETE:
      return [] if flatten else type(self)()
    return res

  def Flatten(self) -> List[Any]:
    """Returns a list containing the flattened values in the `.NestedMap`.

    Unlike py_utils.Flatten(), this will only descend into lists, dicts, and
    NestedMaps and not tuples, or namedtuples.
    """
    return self._RecursiveMap(lambda _, v: v, flatten=True)

  def FlattenItems(self) -> List[Tuple[Any, Any]]:
    """Flatten the `.NestedMap` and returns <key, value> pairs in a list.

    Returns:
      A list of <key, value> pairs, where keys for nested entries will be
      represented in the form of `foo.bar[10].baz`.
    """
    return self._RecursiveMap(lambda k, v: (k, v), flatten=True)

  def Pack(self: NestedMapT, lst: Sequence[Any]) -> NestedMapT:
    """Returns a copy of this with each value replaced by a value in lst."""
    assert len(self.FlattenItems()) == len(lst)
    v_iter = iter(lst)
    return self._RecursiveMap(lambda unused_k, unused_v: next(v_iter))

  def Transform(self: NestedMapT, fn: Callable[[Any], Any]) -> NestedMapT:
    """Returns a copy of this `.NestedMap` with fn applied on each value."""
    return self._RecursiveMap(lambda _, v: fn(v))

  def TransformWithKey(self: NestedMapT, fn: Callable[[str, Any],
                                                      Any]) -> NestedMapT:
    """Returns a copy of this `.NestedMap` with fn applied on each key/value."""
    return self._RecursiveMap(fn)

  def IsCompatible(self, other: NestedMapT) -> bool:
    """Returns true if self and other are compatible.

    If x and y are two compatible `.NestedMap`, `x.Pack(y.Flatten())` produces y
    and vice versa.

    Args:
      other: Another `.NestedMap`.
    """
    items = self._RecursiveMap(lambda k, _: k, flatten=True)
    other_items = other._RecursiveMap(lambda k, _: k, flatten=True)  # pylint: disable=protected-access
    return items == other_items

  def Filter(self: NestedMapT, fn: Callable[[Any], bool]) -> NestedMapT:
    """Returns a copy with entries where fn(entry) is True."""
    return self.FilterKeyVal(lambda _, v: fn(v))

  def FilterKeyVal(self: NestedMapT, fn: Callable[[str, Any],
                                                  bool]) -> NestedMapT:
    """Returns a copy of this `.NestedMap` filtered by fn.

    If fn(key, entry) is True, the entry is copied into the returned NestedMap.
    Otherwise, it is not copied.

    Args:
      fn: a callable of (string, entry)->boolean.

    Returns:
      A `.NestedMap` contains copied entries from this `'.NestedMap`.
    """
    return self._RecursiveMap(lambda k, v: v if fn(k, v) else self._DELETE)

  def _ToStrings(self) -> List[str]:
    """Returns debug strings in a list for this `.NestedMap`."""
    kv = self.FlattenItems()
    maxlen = max([len(k) for k, _ in kv]) if kv else 0
    return sorted([k + ' ' * (4 + maxlen - len(k)) + str(v) for k, v in kv])

  def DebugString(self) -> str:
    """Returns a debug string for this `.NestedMap`."""
    return '\n'.join(self._ToStrings())

  def VLog(self,
           level: Optional[int] = None,
           prefix: Optional[str] = None) -> None:
    """Logs the debug string at the level."""
    if level is None:
      level = 0
    if prefix is None:
      prefix = 'nmap: '
    for l in self._ToStrings():
      tf.logging.vlog(level, '%s %s', prefix, l)

  def __dir__(self) -> List[str]:
    """dir() that includes flattened keys in returned output."""
    keys = self._RecursiveMap(lambda k, v: k, flatten=True)
    return keys + super().__dir__()
