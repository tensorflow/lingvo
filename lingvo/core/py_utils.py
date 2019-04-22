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
"""Common utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import hashlib
import math
import numbers
import re
import traceback

import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.contrib.model_pruning.python.layers import core_layers as pruning_layers
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import function
from tensorflow.python.util import deprecation
from lingvo.core import hyperparams
from lingvo.core import retry
from lingvo.core.ops import py_x_ops

tf.flags.DEFINE_bool('enable_asserts', True,
                     'If False, we disable all asserts.')

tf.flags.DEFINE_bool('enable_check_numerics', True,
                     'If False, we bypass calls to CheckNumerics.')

tf.flags.DEFINE_bool('print_debug_tensors', False,
                     'Whether to print debug tensors.')

tf.flags.DEFINE_string(
    'xla_device', '', 'If non-empty, can be cpu, gpu, or tpu (case sensitive)')

tf.flags.DEFINE_bool('nas_run', False, 'If True, this is a NAS training run.')

tf.flags.DEFINE_bool(
    'use_resource_var', False,
    'Use ResourceVariable instead of Variable; this option is '
    'ignored when xla_device=tpu, as TPU requires resource '
    'variables')

tf.flags.DEFINE_bool(
    'tpu_compatible', False, 'Create variables in a way compatible with TPU. '
    'This should be true for any job that will interact '
    'with variables or a checkpoint that will be produced '
    'or consumed by TPU')

tf.flags.DEFINE_bool(
    'pin_vars_to_cpu', False,
    'Pin variables to cpu:0.  This is useful for weight-sharing / multi-core '
    'inference on TPUs in which TPU core variables are managed via '
    'TPUPartitionedCallOp.')

tf.flags.DEFINE_bool(
    'no_identity_on_vars', False,
    'Do not add tf.identity() on vars. This allows TPUPartitionedCallOp to use'
    'variable handles directly for weight-sharing / multi-core '
    'inference on TPUs.')

FLAGS = tf.flags.FLAGS

ENQUEUE_OPS = '__lingvo_enqueue_ops'
CLOSE_QUEUE_OPS = '__lingvo_close_queue_ops'

# pylint: disable=protected-access
deprecation._PRINT_DEPRECATION_WARNINGS = False

# pylint: enable=protected-access


def Assert(condition, data, *args, **kwargs):
  if FLAGS.enable_asserts:
    return tf.Assert(condition, data, *args, **kwargs)
  else:
    return tf.no_op()


def assert_equal(*args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    return tf.assert_equal(*args, **kwargs)
  else:
    return tf.no_op()


def assert_greater_equal(*args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    return tf.assert_greater_equal(*args, **kwargs)
  else:
    return tf.no_op()


def assert_greater(*args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    return tf.assert_greater(*args, **kwargs)
  else:
    return tf.no_op()


def assert_less_equal(*args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    return tf.assert_less_equal(*args, **kwargs)
  else:
    return tf.no_op()


def assert_less(*args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    return tf.assert_less(*args, **kwargs)
  else:
    return tf.no_op()


def assert_between(x, l, r, *args, **kwargs):  # pylint: disable=invalid-name
  return tf.group(
      Assert(tf.reduce_all(tf.greater_equal(x, l)), [x], *args, **kwargs),
      Assert(tf.reduce_all(tf.less(x, r)), [x], *args, **kwargs))


def assert_shape_match(*args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    filepath, line, func, _ = traceback.extract_stack(limit=3)[-2]
    kwargs['msg'] = 'LINGVO ASSERT %s:%s(%s)' % (re.sub(
        r'.*/', '', filepath), line, func)
    return py_x_ops.assert_shape_match(*args, **kwargs)
  else:
    return tf.no_op()


def assert_same_dim0(xs, *args, **kwargs):  # pylint: disable=invalid-name
  if FLAGS.enable_asserts:
    return py_x_ops.assert_same_dim0(xs, *args, **kwargs)
  else:
    return tf.no_op()


def _CheckNumerics(x, message=None, *args, **kwargs):
  if x.dtype.is_floating:
    if 'name' not in kwargs:
      kwargs['name'] = re.sub(r':\d+', '', x.name) + '_CheckNumerics'
    return tf.check_numerics(x, message if message else x.name, *args, **kwargs)
  else:
    return x


def CheckNumerics(inp, message=None, *args, **kwargs):
  """Check numerics for tensors in inp."""
  if not FLAGS.enable_check_numerics:
    return inp
  if isinstance(inp, list):
    return [_CheckNumerics(x, message, *args, **kwargs) for x in inp]
  if isinstance(inp, tuple):
    return tuple(_CheckNumerics(x, message, *args, **kwargs) for x in inp)
  return _CheckNumerics(inp, message, *args, **kwargs)


def with_dependencies(dependencies, output_tensor):
  with tf.control_dependencies(dependencies):
    return tf.identity(output_tensor)


@contextlib.contextmanager
def _PrintOptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  yield
  np.set_printoptions(**original)


def _Print(name, x):
  with _PrintOptions(linewidth=1000):
    tf.logging.info('%s = %s', name, np.array_repr(x))


def Log(value, prefix, **kwargs):
  """Prints out values of tensors.

  Useful for debugging. E.g.,
    x = ... a tf.Tensor ...
    y = ... a tf.Tensor ...
    z = compute(x, y)
    z = Log(z, 'debug compute()', x=x, y=y)

  Args:
    value: A Tensor. Log happens after this tensor's computed.
    prefix: Every tensor is logged with this prefix.
    **kwargs: keywords and tensors. Tensors are logged in the sort order of
      these keywards.

  Returns:
    value is returned.
  """

  # Ensures tensors are printed in order.
  last = value
  for k in sorted(kwargs):
    with tf.control_dependencies([last]):
      last = tf.py_func(_Print, [prefix + ' : ' + k, kwargs[k]], [])
  with tf.control_dependencies([last]):
    return tf.identity(value)


def _Save(steps, prefix, key, val):
  filename = '%s.%08d.%s.npy' % (prefix.decode(), steps, key.decode())
  with tf.gfile.Open(filename, 'w') as outfile:
    np.save(outfile, val)


def Save(value, filename_prefix, **kwargs):
  """Saves values of tensors into files.

  Useful for debugging. E.g.,
    x = ... a tf.Tensor ...
    y = ... a tf.Tensor ...
    z = compute(x, y)
    z = Save(z, '/path/tmp', x=x, y=y, z=z)

  Args:
    value: A Tensor. Saving happens after this tensor is computed.
    filename_prefix: Every tensor is saved with this filename prefix.
    **kwargs: keywords and tensors. Tensors are logged in the sort order of
      these keywards.

  Returns:
    value is returned.
  """
  last = value
  steps = GetGlobalStep()
  for k in sorted(kwargs):
    with tf.control_dependencies([last]):
      last = tf.py_func(_Save, [steps, filename_prefix, k, kwargs[k]], [])
  with tf.control_dependencies([last]):
    return tf.identity(value)


def HasRank(tensor, expected_rank):
  """Syntactic sugar for asserting that tensor has the expected rank."""
  if tensor.shape.ndims is not None and isinstance(expected_rank, int):
    assert tensor.shape.ndims == expected_rank, (
        'Ranks did not match, got %d, '
        'expected %d') % (tensor.shape.ndims, expected_rank)
    return tensor
  if FLAGS.enable_asserts:
    return with_dependencies([tf.assert_equal(tf.rank(tensor), expected_rank)],
                             tensor)
  else:
    return tensor


def HasShape(tensor, expected_shape):
  """Syntactic sugar for asserting that tensor has the expected shape."""
  if FLAGS.enable_asserts:
    filepath, line, func, _ = traceback.extract_stack(limit=3)[-2]
    msg = 'LINGVO ASSERT %s:%s(%s)' % (re.sub(r'.*/', '',
                                                 filepath), line, func)
    return with_dependencies([
        py_x_ops.assert_shape_match(tf.shape(tensor), expected_shape, msg=msg)
    ], tensor)
  else:
    return tensor


def GetShape(tensor, ndims=None):
  """Returns tensor's shape as a list which can be unpacked, unlike tf.shape.

  Tries to return static shape if it's available. Note that this means
  some of the outputs will be ints while the rest will be Tensors.

  Args:
    tensor: The input tensor.
    ndims: If not None, returns the shapes for the first `ndims` dimensions.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.shape(tensor)
  if ndims is None:
    ndims = tensor.shape.ndims
    if ndims is None:
      return shape
  elif tensor.shape.ndims is not None:
    ndims = min(ndims, tensor.shape.ndims)
  shapes = [
      tensor.shape[x].value if tensor.shape[x].value is not None else shape[x]
      for x in range(ndims)
  ]
  return shapes


def use_xla():  # pylint: disable=invalid-name
  res = FLAGS.xla_device
  if res:
    assert FLAGS.xla_device in ('', 'cpu', 'gpu', 'tpu')
  return res


def use_tpu():  # pylint: disable=invalid-name
  res = FLAGS.xla_device == 'tpu'
  if res:
    assert not FLAGS.enable_asserts  # asserts not supported on tpu
  return res


def nas_run():  # pylint: disable=invalid-name
  return FLAGS.nas_run


def tpu_compat():  # pylint: disable=invalid-name
  return use_tpu() or FLAGS.tpu_compatible


def use_resource_variables():  # pylint: disable=invalid-name
  return FLAGS.use_resource_var or tpu_compat()


@contextlib.contextmanager
def outside_all_rewrites():
  with tf.control_dependencies(None):
    yield


def RunOnTpuHost(func, *args, **kwargs):
  r"""Runs the given function call on TPU host.

  Invokes func(\*args, \*\*kwargs) directly if not running on tpu.

  Args:
    func: the function to invoke.

  Returns:
    The function return value.
  """
  if use_tpu():
    return tpu.outside_compilation(func, *args, **kwargs)
  else:
    return func(*args, **kwargs)


_tpu_device_assignment = None


def SetTpuDeviceAssignment(tpu_device_assignment):
  global _tpu_device_assignment
  if _tpu_device_assignment is not None:
    tf.logging.warning('tpu_device_assignment was already set, '
                       'overwriting with new assignment.')
  _tpu_device_assignment = tpu_device_assignment


# This function should called in unittest only.
def ClearTpuDevice():
  global _tpu_device_assignment
  _tpu_device_assignment = None


def GetTpuDeviceAssignment():
  return _tpu_device_assignment


def SessionConfig(soft_placement=True, inline=True):
  """Returns a session config proto.

  Args:
    soft_placement: Turns allow_soft_placement on iff True.
    inline: Turns do_function_inlining on iff True.

  Returns:
    A TF session config proto.
  """
  session_config = tf.ConfigProto(
      allow_soft_placement=soft_placement,
      graph_options=tf.GraphOptions(
          optimizer_options=tf.OptimizerOptions(
              opt_level=tf.OptimizerOptions.L1, do_function_inlining=inline)))
  # Disable layout optimizer which increases GPU memory usage.
  session_config.graph_options.rewrite_options.layout_optimizer = (
      rewriter_config_pb2.RewriterConfig.OFF)
  return session_config


_NAME_PATTERN = re.compile('[A-Za-z_][A-Za-z0-9_]*')


class NestedMap(dict):
  """A simpler helper to maintain a dict.

  E.g.::

      >>> foo = NestedMap()
      >>> foo['x'] = 10
      >>> foo.y = 20
      >>> assert foo.x * 2 == foo.y
  """

  # Disable pytype attribute checking.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, *args, **kwargs):
    super(NestedMap, self).__init__(*args, **kwargs)
    self.__dict__ = self

  def __setitem__(self, key, value):
    NestedMap.CheckKey(key)
    super(NestedMap, self).__setitem__(key, value)

  def __getattribute__(self, key):
    """Wrapper to help user know what available attributes are."""
    try:
      return super(NestedMap, self).__getattribute__(key)
    except AttributeError as e:
      raise AttributeError('%s; available attributes: %s' %
                           (e, self.__dict__.keys()))

  def copy(self):  # Don't delegate w/ super: dict.copy() -> dict.
    return NestedMap(self)

  def __deepcopy__(self, unused_memo):
    return self.DeepCopy()

  def DeepCopy(self):
    flat_v = self.Flatten()
    return self.Pack(flat_v)

  @staticmethod
  def CheckKey(key):
    """Asserts that key is valid NestedMap key."""
    assert isinstance(key, six.string_types) and _NAME_PATTERN.match(key), key

  def Flatten(self):
    """Flatten the `.NestedMap` and returns values in a list."""

    def Expand(v):
      if isinstance(v, NestedMap):
        return v.Flatten()
      elif isinstance(v, list):
        ret = []
        for x in v:
          ret += Expand(x)
        return ret
      else:
        return [v]

    ret = []
    for k in sorted(self.keys()):
      ret += Expand(self[k])
    return ret

  def FlattenItems(self):
    """Flatten the `.NestedMap` and returns <key, value> pairs in a list.

    For lists, keys will be returned with `_<idx>` appended, e.g. `x.y_10.z`.

    Returns:
      A list of <key, value> pairs, where keys for nested entries will be
      represented in the form of `foo.bar`.
    """

    def Expand(key, v):
      if isinstance(v, NestedMap):
        ret = []
        for k in sorted(v.keys()):
          global_key = key + '.' + k if key else k
          ret += Expand(global_key, v[k])
        return ret
      elif isinstance(v, list):
        ret = []
        for i, x in enumerate(v):
          ret += Expand('%s_%d' % (key, i), x)
        return ret
      else:
        return [(key, v)]

    return Expand(None, self)

  def Transform(self, fn):
    """Returns a copy of this `.NestedMap` with fn applied on each value."""

    def DoTransform(v):
      if isinstance(v, NestedMap):
        return v.Transform(fn)
      elif isinstance(v, list):
        ret = []
        for x in v:
          ret += [DoTransform(x)]
        return ret
      else:
        return fn(v)

    ret = NestedMap()
    for k in sorted(self.keys()):
      ret[k] = DoTransform(self[k])

    return ret

  def Filter(self, fn):
    """Returns a copy of this `.NestedMap` with entries that fn(entry) is True."""
    return self.FilterKeyVal(lambda _, v: fn(v))

  def FilterKeyVal(self, fn):
    """Returns a copy of this `.NestedMap` with filtered by fn.

    If fn(key, entry) is True, the entry is copied into the returned NestedMap.
    Otherwise, it is not copied.
    For lists, keys will be processed with indices, e.g. `x.y[10].z`.
    This is different from FlattenItems.

    Args:
      fn: a callable of (string, entry)->boolean.

    Returns:
      A `.NestedMap` contains copied entries from this `'.NestedMap`.
    """

    def DoFilter(prefix, value):
      """Recursively copy value with the filter fn applied."""
      if isinstance(value, NestedMap):
        ret = NestedMap()
        for k in sorted(value.keys()):
          v = value[k]
          if prefix:
            key = '%s.%s' % (prefix, k)
          else:
            key = k
          filtered = DoFilter(key, v)
          if filtered is not None:
            ret[k] = filtered
        return ret if len(ret) else None
      elif isinstance(value, list):
        lst = []
        for i, x in enumerate(value):
          filtered = DoFilter('%s[%d]' % (prefix, i), x)
          if filtered is not None:
            lst += [filtered]
        return lst if lst else None
      elif fn(prefix, value):
        return value
      else:
        return None

    return DoFilter('', self)

  def Pack(self, lst):
    """Returns a copy of this with each value replaced by a value in lst."""

    class DoPack(object):

      def __init__(self, lst):
        self._iter = iter(lst)

      def __call__(self, x):
        del x
        return next(self._iter)

    return self.Transform(DoPack(lst))

  def IsCompatible(self, other):
    """Returns true if self and other is compatible.

    Args:
      other: Another `.NestedMap`.  If x and y are two compatible `.NestedMap`,
        `x.Pack(y.Flatten())` produces y and `y.Pack(x.Flatten())` produces x.
    """

    def DoCompare(x, y):
      """Compares x and y."""
      if isinstance(x, NestedMap) or isinstance(y, NestedMap):
        if not isinstance(x, NestedMap) or not isinstance(y, NestedMap):
          return False
        if sorted(x.keys()) != sorted(y.keys()):
          return False
        for (k, v) in six.iteritems(x):
          if not DoCompare(v, y[k]):
            return False
      elif isinstance(x, list) or isinstance(y, list):
        if not isinstance(x, list) or not isinstance(y, list):
          return False
        if len(x) != len(y):
          return False
        for (u, v) in zip(x, y):
          if not DoCompare(u, v):
            return False
      return True

    return DoCompare(self, other)

  def DebugString(self):
    """Returns a debug string for this `.NestedMap`."""

    def Print(prefix, value):
      """Recursively walk value."""
      ret = []
      if isinstance(value, NestedMap):
        for k, v in six.iteritems(value):
          if prefix:
            key = '%s.%s' % (prefix, k)
          else:
            key = k
          ret += Print(key, v)
      elif isinstance(value, list):
        for i, x in enumerate(value):
          ret += Print('%s[%d]' % (prefix, i), x)
      else:
        ret += [(prefix, value)]
      return ret

    kv = Print('', self)
    maxlen = max([len(k) for k, _ in kv]) if kv else 0
    strs = [k + ' ' * (4 + maxlen - len(k)) + str(v) for k, v in kv]
    return '\n'.join(sorted(strs))


class _Unique(object):
  """A helper to uniqify variables in a NestedMap."""

  def __init__(self):
    self._vset = set()

  def __call__(self, v):
    if (v is None) or (v in self._vset):
      return False
    else:
      self._vset.add(v)
      return True


def ToUniqueList(nmap):
  """Returns the flattened `nmap` with duplicates removed."""
  return nmap.Filter(_Unique()).Flatten()


def ReadOnlyAttrDictView(backing):
  """Wraps a dict to provide a read-only view of its contents.

  Dict keys can also be accessed by attribute.

  Args:
    backing: Dict-like object to wrap.

  Returns:
    Read-only Mapping that can be accessed by index (['foo']) or attr (d.foo).
  """

  class Wrapper(object):
    """Wrapper object."""

    # Disable pytype attribute checking.
    _HAS_DYNAMIC_ATTRIBUTES = True

    def __getitem__(self, key):
      return backing[key]

    def __len__(self):
      return len(backing)

    def __iter__(self):
      return iter(backing)

    def __getattr__(self, key):
      return backing[key]

    def __hasattr__(self, key):
      return key in backing

    def __setattr__(self, key, value):
      raise AttributeError('Dictionary is read-only.')

    def __setitem__(self, key, value):
      raise AttributeError('Dictionary is read-only.')

  return Wrapper()


class RNNCellStateInit(object):
  """State initialization functions for RNN cell init state."""

  @staticmethod
  def _Params(method, seed):
    p = hyperparams.Params()
    p.Define('method', method,
             'Initialization method. Should be one of zeros, random_normal.')
    p.Define('seed', seed, 'Random seed used to generate initial values.')
    p.Freeze()
    return p

  @staticmethod
  def Zeros():
    """tf.zeros()."""
    return RNNCellStateInit._Params('zeros', seed=None)

  @staticmethod
  def RandomNormal(seed=None):
    """tf.random.normal()."""
    return RNNCellStateInit._Params('random_normal', seed)


def DefaultRNNCellStateInit():
  return RNNCellStateInit.Zeros()


def InitRNNCellState(shape, init=None, dtype=None, name=None):
  """Initial state definitions for RNN cell implementations.

  Args:
    shape: A array of ints for specifying the shape of the state.
    init: Hyperparameters as returned by one of the static implemetaitons in
      RNNCellStateInit.
    dtype: The dype of the states. Defaults to tf.float32.
    name: An optional name for the operation.

  Returns:
    A Tensor of the specified shape, and sampled from the distribution as
    defined by the init parameters.
  """
  if init is None:
    init = DefaultRNNCellStateInit()
  if dtype is None:
    dtype = tf.float32

  method = init.method
  if method in ['zeros']:
    init_state = tf.zeros(shape=shape, dtype=dtype, name=name)
  elif method in ['random_normal']:
    init_state = tf.random.normal(
        shape=shape, dtype=dtype, name=name, seed=init.seed)
  else:
    raise ValueError('zero_state method (%s) not supported.' % method)

  return init_state


class WeightInit(object):
  """Static class providing weight initialization config params."""

  @staticmethod
  def _Params(method, scale, seed):
    p = hyperparams.Params()
    p.Define('method', method, 'Initialization method.')
    p.Define('scale', scale, 'Initialization scale.')
    p.Define('seed', seed, 'Random seed used to generate initial values.')
    p.Freeze()
    return p

  @staticmethod
  def Gaussian(scale=1.0, seed=None):
    """scale * tf.random_normal(0, 1.0)."""
    return WeightInit._Params('gaussian', scale, seed)

  @staticmethod
  def Uniform(scale=1.0, seed=None):
    """scale * tf.random_uniform(-1.0, 1.0)."""
    return WeightInit._Params('uniform', scale, seed)

  @staticmethod
  def UniformPositive(scale=1.0, seed=None):
    """scale * tf.random_uniform(0., 1.0)."""
    return WeightInit._Params('uniform_positive', scale, seed)

  @staticmethod
  def Xavier(scale=1.0, seed=None):
    """Xavier initialization (x = sqrt(6. / (in + out)); [-x, x])."""
    return WeightInit._Params('xavier', scale, seed)

  @staticmethod
  def GeoMeanXavier(scale=1.0, seed=None):
    """A variant of Xavier (x = sqrt(3. / sqrt(in * out)); [-x, x])."""
    return WeightInit._Params('geo_mean_xavier', scale, seed)

  @staticmethod
  def Constant(scale=1.0):
    """scale."""
    return WeightInit._Params('constant', scale, 0)

  @staticmethod
  def TruncatedGaussian(scale=1.0, seed=None):
    """scale * tf.truncated_normal(0, 1.0)."""
    return WeightInit._Params('truncated_gaussian', scale, seed)

  @staticmethod
  def GaussianSqrtDim(scale=1.0, seed=None):
    """scale * tf.random_normal(0, 1 / sqrt(dim0))."""
    return WeightInit._Params('gaussian_sqrt_dim', scale, seed)

  @staticmethod
  def GaussianSqrtFanIn(scale=1.0, seed=None):
    """scale * tf.random_normal(0, 1 / sqrt(fan_in))."""
    return WeightInit._Params('gaussian_sqrt_fanin', scale, seed)

  @staticmethod
  def GaussianSqrtFanOut(scale=1.0, seed=None):
    """scale * tf.random_normal(0, 1 / sqrt(fan_out))."""
    return WeightInit._Params('gaussian_sqrt_fanout', scale, seed)

  @staticmethod
  def UniformSqrtDim(scale=1.0, seed=None):
    """scale * tf.uniform(-1 / sqrt(dim0), 1 / sqrt(dim0))."""
    return WeightInit._Params('uniform_sqrt_dim', scale, seed)

  @staticmethod
  def UniformUnitScaling(scale=1.0, seed=None):
    """scale * sqrt(3) / sqrt(dim0) * tf.uniform(-1, 1)."""
    return WeightInit._Params('uniform_unit_scaling', scale, seed)

  @staticmethod
  def TruncatedGaussianSqrtDim(scale=1.0, seed=None):
    """scale * tf.truncated_normal(0, 1 / sqrt(dim0))."""
    return WeightInit._Params('truncated_gaussian_sqrt_dim', scale, seed)

  @staticmethod
  def TruncatedGaussianSqrtFanIn(scale=1.0, seed=None):
    """scale * tf.truncated_normal(0, 1 / sqrt(fan_in))."""
    return WeightInit._Params('truncated_gaussian_sqrt_fanin', scale, seed)

  @staticmethod
  def TruncatedGaussianSqrtFanOut(scale=1.0, seed=None):
    """scale * tf.truncated_normal(0, 1 / sqrt(fan_out))."""
    return WeightInit._Params('truncated_gaussian_sqrt_fanout', scale, seed)

  @staticmethod
  def KaimingUniformFanInRelu(scale=1.0, seed=None):
    return WeightInit._Params('kaiming_uniform_fanin_relu', scale, seed)

  @staticmethod
  def KaimingUniformFanInLeakyRelu(scale=np.sqrt(5.), seed=None):
    return WeightInit._Params('kaiming_uniform_fanin_leakyrelu', scale, seed)


_DEFAULT_XAVIER_INIT = 1.000001


def DefaultParamInit():
  # Here we use 1.000001 as a signature for user picking up the
  # default param initializer.
  return WeightInit.Xavier(_DEFAULT_XAVIER_INIT)


def IsDefaultParamInit(p):
  return (p.method == 'xavier' and p.scale == _DEFAULT_XAVIER_INIT and
          p.seed is None)


def WeightParams(shape, init=None, dtype=None, collections=None):
  """Returns a hyperparams for a weight variable given the shape/init/dtype."""
  if init is None:
    init = WeightInit.Xavier(_DEFAULT_XAVIER_INIT)
  if dtype is None:
    dtype = tf.float32
  if collections is None:
    collections = []
  p = hyperparams.Params()
  p.Define('dtype', dtype, 'The weight data type.')
  p.Define('shape', shape, 'The weight shape.')
  p.Define('init', init, 'Initialization method.')
  p.Define('collections', collections,
           'Variable collections this weight belongs to.')
  p.Freeze()
  return p


def FindNeeded(endpoints):
  """List names of tensors and operations required to compute endpoints."""
  names_seen = set()
  queue = []
  for e in tf.contrib.framework.nest.flatten(endpoints):
    if isinstance(e, tf.Operation):
      queue.append(e)
    else:
      queue.append(e.op)
  while queue:
    op = queue.pop()
    name = op.name
    if name not in names_seen:
      names_seen.add(name)
      names_seen.update((o.name for o in op.outputs))
      queue.extend(i.op for i in op.inputs)
      queue.extend(op.control_inputs)
  return names_seen


def FindNeededInList(tensor_list, endpoints):
  """Return tensors from tensor_list needed to compute any of endpoints."""
  all_needed = FindNeeded(endpoints)
  return [t for t in tensor_list if t.name in all_needed]


class _CollectionGetter(object):
  """Get graph local value from a defined collection."""

  def __init__(self, key, default_factory):
    self._key = key
    self._default_factory = default_factory

  def __call__(self):
    collection = tf.get_collection(self._key)
    if collection:
      assert len(collection) == 1
      return collection[0]
    value = self._default_factory()
    tf.add_to_collection(self._key, value)
    return value


# Global variable to control multitask variable reuse
# If False (default) the default tf.get_variable is used, that is:
# - Reusing scopes only allow getting existing variables
# - Non-reusing scopes only allow getting new variables
# With OPPORTUNISTIC_VARIABLE_REUSE==True:
# - Reusing scopes only allow getting existing variables, as usual
# - Non-reusing scopes reuse new variables or get new ones
_OPPORTUNISTIC_VARIABLE_REUSE_KEY = ('__lingvo_opportunistic_variable_reuse',)

_get_opportunistic_variable_reuse = _CollectionGetter(
    _OPPORTUNISTIC_VARIABLE_REUSE_KEY, lambda: [False])

_VARIABLE_RENAME_RULES_KEY = ('__lingvo_variable_rename_rules',)

_get_rename_rules_stack = _CollectionGetter(
    _VARIABLE_RENAME_RULES_KEY, lambda: [])


@contextlib.contextmanager
def OpportunisticVariableReuseScope(enable_opportunistic_reuse=True):
  opportunistic_var_reuse = _get_opportunistic_variable_reuse()
  old_val = opportunistic_var_reuse[0]
  opportunistic_var_reuse[0] = enable_opportunistic_reuse
  yield
  opportunistic_var_reuse[0] = old_val


@contextlib.contextmanager
def VariableRenameScope(renames):
  """Append the renaming rules to the stack of renames.

  Args:
    renames: pairs of (regexp, new_name_format). If the regexp matches, the
      new_name_format will be interpolated using the matched groups.

  Yields:
    scope in which the renaming rules are applied
  """
  rename_rules_stack = _get_rename_rules_stack()
  rename_rules_stack.append(renames)
  yield
  rename_rules_stack.pop()


def GetVariableName(name):
  """Get variable name after application of all renaming rules.

  Args:
    name: untransformed variable name with scope_name prepended

  Returns:
    name possibly modified using renaming rules
  """
  matched = False
  new_name = name
  for renames in _get_rename_rules_stack():
    for regexp, name_format in renames:
      match = re.match(regexp, name)
      if match:
        if matched:
          tf.logging.warn('Multiple matches for: %s', name)
        matched = True
        new_name = name_format % match.groups()
  if new_name != name:
    tf.logging.info("WARNING!!! Renaming variable '%s' to '%s'", name, new_name)
  return new_name


def GenerateSeedFromName(name):
  """Generate a random seed from a name string."""
  md5 = hashlib.md5()
  md5.update(name.encode('utf-8'))
  return int(md5.hexdigest(), 16) % (2**31 - 1)


# To keep track of all the variables ever gets created by the CreateVariable
# routine below.
_ALL_VARS_KEY = ('__lingvo_all_vars',)

_get_all_vars = _CollectionGetter(_ALL_VARS_KEY, lambda: {})


def GetFanInFanOut(shape):
  """Returns (fan_in, fan_out) of a weight variable of the give shape."""
  if not shape:
    return None, None
  if len(shape) < 1:
    return 1, 1
  elif len(shape) == 1:
    # Following _compute_fans() from TF's init_ops.py.
    return shape[0], shape[0]
  else:
    receptive_field_size = 1
    for s in shape[:-2]:
      receptive_field_size *= s
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


# TODO(yonghui): Add support for partitioned Variables.
def CreateVariable(name,
                   params,
                   reuse=None,
                   trainable=True,
                   init_wrapper=None,
                   collections=None):
  """Creates tf.Variable according to param_config.

  Args:
    name: A string, name of the variable.
    params: A WeightParams specifying the details of how this variable should be
      constructed and initialized.
    reuse: Whether or not to reuse an existing variable. It has the same
      semantics as the reuse arg in tf.variable_scope.
    trainable: Whether or not the variable is trainable.
    init_wrapper: a callback which takes a tf initializer callable and returns a
      tensor. It is used when shape of the variable isn't statically
      determinable.
    collections: Override the default variable collection (
      tf.GraphKeys.GLOBAL_VARIABLES).

  Returns:
    tf.identity(var), var pair. The tf.identity() node is colocated
    with var. In the case of FLAGS.no_identity_on_vars, simply returns
    a var, var pair.
  """
  p = params.Copy()
  assert isinstance(p, hyperparams.Params)
  dtype = p.dtype
  shape = p.shape
  dim0 = 1
  if shape:
    assert all([dim_size > 0 for dim_size in shape]), ('%s' % shape)
    dim0 = shape[0]
  assert np.all(p.init.scale >= 0) or p.init.method == 'constant'
  method = p.init.method
  scale = p.init.scale
  seed = p.init.seed

  if IsDefaultParamInit(p.init):
    tf.logging.warning(
        'WARNING!!! var %s is using the default xavier initializer.'
        ' Make sure this is intended.', name)

  if tf.get_default_graph().seed is not None:
    # We are in a program/test which need determistic randomization.
    if seed is None:
      # We are not given a per-variable random seed. We use hash of
      # variable name as a stable random seed.
      with tf.variable_scope(name) as scope:
        var_name = GetVariableName(scope.name)
      seed = GenerateSeedFromName(var_name)

  if (method in [
      'gaussian_sqrt_dim', 'uniform_sqrt_dim', 'truncated_gaussian_sqrt_dim'
  ]):
    if len(shape) > 2:
      # This is probably not the right method to use when len(shape) > 2,
      # e.g. dim0 will be 3 with a 3x3 conv2d kernel.
      tf.logging.warn(
          'Initializing %s of shape %s with method %s: dim0=%s. '
          'Make sure that it is intended.', name, shape, method, dim0)
    scale *= 1.0 / math.sqrt(dim0)

  if method in ['gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanin']:
    fan_in, _ = GetFanInFanOut(shape)
    if fan_in is not None:
      scale *= 1.0 / math.sqrt(fan_in)
  if method in ['gaussian_sqrt_fanout', 'truncated_gaussian_sqrt_fanout']:
    _, fan_out = GetFanInFanOut(shape)
    if fan_out is not None:
      scale *= 1.0 / math.sqrt(fan_out)

  init_dtype = dtype.real_dtype
  if method in [
      'gaussian', 'gaussian_sqrt_dim', 'gaussian_sqrt_fanin',
      'gaussian_sqrt_fanout'
  ]:
    v_init = tf.random_normal_initializer(
        mean=0.0, stddev=scale, seed=seed, dtype=init_dtype)
  elif method in ['uniform', 'uniform_sqrt_dim']:
    v_init = tf.random_uniform_initializer(
        minval=-scale, maxval=scale, seed=seed, dtype=init_dtype)
  elif method in ['uniform_positive']:
    v_init = tf.random_uniform_initializer(
        minval=0.0, maxval=scale, seed=seed, dtype=init_dtype)
  elif method in ['uniform_unit_scaling']:
    v_init = tf.uniform_unit_scaling_initializer(
        factor=scale, seed=seed, dtype=init_dtype)
  elif method in [
      'truncated_gaussian', 'truncated_gaussian_sqrt_dim',
      'truncated_gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanout'
  ]:
    v_init = tf.truncated_normal_initializer(
        mean=0.0, stddev=scale, seed=seed, dtype=init_dtype)
  elif method in ['constant']:
    v_init = tf.constant_initializer(value=scale, dtype=init_dtype)
  elif method in ['xavier', 'geo_mean_xavier']:
    # pylint: disable=unused-argument
    def XavierUniform(shape, dtype, partition_info):
      """Xavier initialization (x = sqrt(6. / (in + out)); scale*[-x, x])."""
      if not shape:
        raise ValueError(
            '\'shape\' must not be \'None\' or 0 for XavierUniform')
      fan_in, fan_out = GetFanInFanOut(shape)
      if method == 'xavier':
        limit = math.sqrt(6. / (fan_in + fan_out))
      elif method == 'geo_mean_xavier':
        limit = math.sqrt(3. / math.sqrt(fan_in * fan_out))
      return scale * tf.random_uniform(shape, -limit, limit, dtype, seed)

    # pylint: enable=unused-argument
    v_init = XavierUniform
  elif method in [
      'kaiming_uniform_fanin_relu', 'kaiming_uniform_fanin_leakyrelu'
  ]:
    fan_in = np.prod(shape[:-1])
    if method == 'kaiming_uniform_fanin_leakyrelu':
      # Assume the 'a' parameter is the 'scale' argument.
      gain = np.sqrt(2. / (1 + scale**2))
    else:
      gain = np.sqrt(2.)
    std_dev = gain / np.sqrt(fan_in)
    bound = np.sqrt(3.0) * std_dev
    v_init = tf.random_uniform_initializer(
        minval=-bound, maxval=bound, seed=seed, dtype=init_dtype)
  else:
    assert False, 'init_type not supported.'
  if init_wrapper:
    assert shape is None, (
        'Expecting \'params.shape\' being None when '
        '\'init_wrapper\' is specified, instead getting %s') % p.shape
    # Later variable will init from Tensor value instead of intializer callable.
    v_init = init_wrapper(init_dtype, v_init)
  if dtype == tf.complex64:

    def ComplexWrapper(init):

      def _Wrapper(shape, dtype, partition_info):
        # A more complex alternative may be to use the init function for
        # magnitudes and uniform random for phases instead.
        shape = [2] + shape
        value = init(shape, init_dtype, partition_info)
        return tf.complex(value[0], value[1])

      return _Wrapper

    v_init = ComplexWrapper(v_init)

  # TODO(yonghui): Possibly get away from variable_scope and implement our own
  # variable sharing mechanism.
  def GetVar(reuse=reuse):
    """reuse: Whether to reuse the variables."""
    with tf.variable_scope(name) as scope:
      var_name = GetVariableName(scope.name)
      var_scope = tf.VariableScope(
          scope.reuse,
          custom_getter=scope.custom_getter,
          caching_device=scope.caching_device,
          use_resource=scope.use_resource or use_resource_variables())
    with tf.variable_scope(var_scope), \
        tf.variable_scope(var_name, reuse=reuse) as scope:
      if FLAGS.pin_vars_to_cpu:
        with tf.device('/cpu:0'):
          return tf.get_variable(
              'var',
              shape,
              dtype,
              v_init,
              collections=collections,
              trainable=trainable,
              validate_shape=True if shape is not None else False)
      else:
        return tf.get_variable(
            'var',
            shape,
            dtype,
            v_init,
            collections=collections,
            trainable=trainable,
            validate_shape=True if shape is not None else False)

  if _get_opportunistic_variable_reuse()[0]:
    try:
      var = GetVar()
    except ValueError:  # Possibly the variable already exists
      var = GetVar(reuse=True)
  else:
    var = GetVar()

  all_vars = _get_all_vars()
  if var in all_vars:
    tf.logging.info('Reusing var %s', var.name)
    cached = all_vars[var]
    assert cached == p, ('Cached config:\n %s vs new config:\n %s' %
                         (cached.ToText(), p.ToText()))
  else:
    tf.logging.info('Creating var %s shape=%s on device %s', var.name,
                    var.shape, var.device)
    all_vars[var] = p.Copy()
    for col in p.collections:
      tf.add_to_collection(col, var)

  if FLAGS.no_identity_on_vars:
    with tf.device(var.device):
      return var, var
  else:
    # This tf.identity colocated with var.
    with tf.device(var.device):
      return tf.identity(var), var


global_variable_scope = tf.get_variable_scope()

_GLOBAL_STEP_STACK = []


@contextlib.contextmanager
def GlobalStepContext(global_step_tensor):
  _GLOBAL_STEP_STACK.append(global_step_tensor)
  yield
  _GLOBAL_STEP_STACK.pop()


def GetGlobalStep():
  """Return the global_step."""
  if _GLOBAL_STEP_STACK:
    return _GLOBAL_STEP_STACK[-1]
  return tf.train.get_global_step()


def GetOrCreateGlobalStepVar():
  """Return the global_step variable, creating it if it does not exist.

  Prefer GetGlobalStep if a tensor rather than a tf.Variable is sufficient.
  """
  with tf.variable_scope(
      global_variable_scope, use_resource=use_resource_variables()):
    return tf.train.get_or_create_global_step()


def LogMultiLines(label, lines):
  if not isinstance(lines, (list, tuple)):
    lines = lines.split('\n')
  for line in lines:
    tf.logging.info('%s: %s', label, line)


def _LogPlacement(label, theta, copy):
  """Logs theta and its copy's device placement."""

  def GetDevices(m):
    """Flatten a `.NestedMap` m and extracts each value's device."""
    return [x.device for x in m.Flatten()]

  tf.logging.info('=== %s ===', label)
  LogMultiLines(
      label,
      theta.Pack([('%s -> %s' % (x[0], x[1]))
                  for x in zip(GetDevices(theta), GetDevices(copy))
                 ]).DebugString())
  tf.logging.info('==========')


def CreateLocalTheta(theta, device_list=None, label=None):
  """Creates local copy of theta and shards across devices device list.

  Leaves variables intact.

  Args:
    theta: a `.NestedMap` of variables.
    device_list: list of devices to shard across. If None, defaults to a list
      [''].
    label: Logging label.

  Returns:
    A `.NestedMap` of identity() wrapped theta
  """

  class AddIdentity(object):

    def __init__(self, device_list):
      self._list = device_list if device_list else ['']
      self._index = 0

    def __call__(self, x):
      if isinstance(x, tf.Variable):
        return x
      with tf.device(self._list[self._index % len(self._list)]):
        self._index += 1
        return tf.identity(x)

  copy = theta.Transform(AddIdentity(device_list))
  _LogPlacement(label, theta, copy)
  return copy


def _GetVarsToLoad(all_vars, variable_loading_rules, var_ignore_rules):
  """Determines variables to load and their names in checkpoint."""
  # This list contains mappings from var names as they appear in the checkpoint
  # to the vars in our model they correspond to.
  vars_to_load = []
  for model_var in all_vars:
    already_matched = False
    for regexp, name_format in variable_loading_rules:
      match = re.match(regexp, model_var.name)
      # Skip if var doesn't match the loading rules, or if it should be ignored.
      if not match or any(
          re.match(r, model_var.name) for r in var_ignore_rules):
        continue
      assert not already_matched, '%s is already matched!' % model_var.name
      already_matched = True
      checkpoint_var_name = name_format % match.groups()
      if checkpoint_var_name.endswith(':0'):
        checkpoint_var_name = checkpoint_var_name[:-2]
      tf.logging.info('Loading %s from %s', model_var, checkpoint_var_name)
      vars_to_load.append((checkpoint_var_name, model_var))
  return vars_to_load


def _OverrideVarsFromCheckpoint(sess, all_vars, checkpoint_path,
                                variable_loading_rules, var_ignore_rules):
  """Overrides variables from a provided checkpoint."""
  vars_to_load = _GetVarsToLoad(all_vars, variable_loading_rules,
                                var_ignore_rules)
  if not vars_to_load:
    raise ValueError(('Variable loading rules did not match any vars. '
                      'All known: %r') % [v.name for v in all_vars])
  load_var_names = sorted([v.name for _, v in vars_to_load])
  tf.logging.info('Overriding vars from checkpoint: %r', load_var_names)

  while vars_to_load:
    # When restoring, it's possible the same value in the checkpoint
    # can be restored to multiple variables (e.g. during
    # distillation).  However, tf.train.Saver, since it's used for
    # both saving and restoring, requires the name in the checkpoint
    # to be unique for each variable.  So, we call it multiple times
    # with a unique set of names each time.
    unique_vars_to_load = {}
    remaining_vars_to_load = []
    for k, v in vars_to_load:
      if k not in unique_vars_to_load:
        unique_vars_to_load[k] = v
      else:
        remaining_vars_to_load.append((k, v))
    tf.train.Saver(var_list=unique_vars_to_load).restore(sess, checkpoint_path)
    vars_to_load = remaining_vars_to_load


def OverrideVarsFromCheckpoints(session, all_vars, ckpts_loading_rules):
  """Overrides model variables from checkpoints.

  Args:
    session: Tensorflow session.
    all_vars: List of all the parameters in the model.
    ckpts_loading_rules: A dictionary of checkpoint path: loading rules.
      Checkpoint path must be a path to a pretrained model, and loading rules is
      expected to be a tuple of two lists. The first consisting of tuples of
      strings defining (regex to match parameter names in the model to override,
      format string to determine the corresponding var in the checkpoint), and
      the second list consisting of a list of regexes to match parameter names
      in the model which should not be overridden, even if they match those in
      the loading rules.

  Raises:
    ValueError: if colliding vars exist or loading rules is not a list.
  """
  if len(ckpts_loading_rules) > 1:
    tf.logging.info('Overriding vars from multiple checkpoints.')

  vars_overridden = set()
  for ckpt_path, loading_rules in ckpts_loading_rules.items():
    tf.logging.info('Overriding vars from checkpoint: %s', ckpt_path)

    if not isinstance(loading_rules, tuple):
      raise ValueError('Loading rules for %s must be a tuple of two lists!' %
                       ckpt_path)
    if len(loading_rules) != 2 or not all(
        isinstance(l, list) for l in loading_rules):
      raise ValueError('Loading rules for %s must be a tuple of two lists!' %
                       ckpt_path)

    # Filter the model variables to be overridden.
    vars_to_override = [
        var[1]
        for var in _GetVarsToLoad(all_vars, loading_rules[0], loading_rules[1])
    ]

    overlap = set.intersection(vars_overridden, vars_to_override)
    if overlap:
      raise ValueError('Colliding variables to override: %s' % overlap)

    _OverrideVarsFromCheckpoint(session, all_vars, ckpt_path, loading_rules[0],
                                loading_rules[1])
    vars_overridden.update(vars_to_override)
  tf.logging.info('Model variables overridden: %s', vars_overridden)


def _ComputeGradientsSimple(loss, all_vars, grad_aggregation_method,
                            colocate_gradients_with_ops, gate_gradients):
  return tf.gradients(
      loss,
      all_vars,
      aggregation_method=grad_aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      gate_gradients=gate_gradients)


def _ComputeGradientsTpu(loss, all_vars, grad_aggregation_method,
                         colocate_gradients_with_ops, gate_gradients):
  """Computes gradients for local loss across whole TPU cluster."""
  # Scale the loss to account for the full batch size.
  shards = tpu_function.get_tpu_context().number_of_shards
  assert shards
  loss *= tf.constant(1.0 / shards, dtype=loss.dtype)

  # Computes the gradients.
  # Sum the grads so that we can compute statistics across the whole batch.
  all_grads = _ComputeGradientsSimple(loss, all_vars, grad_aggregation_method,
                                      colocate_gradients_with_ops,
                                      gate_gradients)

  # NOTE: We can't use tpu_optimizer.CrossShardOptimizer since
  # we need to scale the grads *after* the cross_replica_sum to
  # match GPU version!
  # TODO(cwhipkey): should we do something different here? - we could do
  # some operations on the gradients before the aggregation (see comments in
  # tensorflow/contrib/tpu/python/tpu/tpu_optimizer.py - see compute_gradients -
  # for some more details).
  aggregated_grads = []
  for g in all_grads:
    if g is not None:
      with tf.colocate_with(g):
        aggregated_grads.append(tf.contrib.tpu.cross_replica_sum(g))
    else:
      aggregated_grads.append(None)
  return aggregated_grads


def _ComputeGradientsTpuNas(loss, all_vars, grad_aggregation_method,
                            colocate_gradients_with_ops, gate_gradients):
  """Computes gradients for local loss across whole TPU cluster.

  This implementation specializes for the case where weight params maybe used
  for different number of times in the forward computation, so that gradients
  should be normalized by the actual number of times they are being computed.

  TODO(yonghui): Maybe merge this implementation with the _ComputeGradientsTpu
  one.

  Args:
    loss: The loss to backprop from.
    all_vars: Vars with respect to which gradients are to be computed.
    grad_aggregation_method: aggregation method to use when calling
      tf.gradients.
    colocate_gradients_with_ops: boolean, whether or not to colocate gradient op
      with the original op.
    gate_gradients: boolean, flag to be passed to tf.gradients.

  Returns:
    gradients to be passed back.
  """
  # Computes the gradients.
  # Sum the grads so that we can compute statistics across the whole batch.
  all_grads = _ComputeGradientsSimple(loss, all_vars, grad_aggregation_method,
                                      colocate_gradients_with_ops,
                                      gate_gradients)

  # NOTE: We can't use tpu_optimizer.CrossShardOptimizer since
  # we need to scale the grads *after* the cross_replica_sum to
  # match GPU version!

  # TODO(cwhipkey): should we do something different here? - we could do
  # some operations on the gradients before the aggregation (see comments in
  # tensorflow/contrib/tpu/python/tpu/tpu_optimizer.py - see compute_gradients -
  # for some more details).

  aggregated_grads = []
  for g in all_grads:
    if g is not None:
      with tf.colocate_with(g):
        # Q(yonghui): Is there a better way to detect a non-zero gradient?
        # Note(yonghui): gradient of a weight param can be all zero if that
        # weight param is not used in the forward computation, e.g. as in
        # switchable layers in neural architecture search.
        zero_threashold = 1e-8
        g_is_non_zero = tf.cast(
            tf.reduce_sum(tf.math.abs(g)) > zero_threashold, g.dtype)
        num_updates = tf.maximum(
            tf.contrib.tpu.cross_replica_sum(g_is_non_zero), 1.0)
        normalized_g = tf.contrib.tpu.cross_replica_sum(g) / num_updates
        aggregated_grads.append(normalized_g)
    else:
      aggregated_grads.append(None)
  return aggregated_grads


def ComputeGradients(
    loss,
    vmap,
    grad_aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
    colocate_gradients_with_ops=True,
    gate_gradients=False):
  """Computes gradients of variables in vmap w.r.t loss.

  Args:
    loss: A scalar Tensor.
    vmap: A `.NestedMap` of variables.
    grad_aggregation_method: Specifies the method used to combine gradient
      terms. Accepted values are constants defined in the class
      AggregationMethod.
    colocate_gradients_with_ops: If True, try colocating gradients with the
      corresponding op.
    gate_gradients: If True, add a tuple around the gradients returned for an
      operations. This avoids some race conditions.

  Returns:
    var_grad - a `.NestedMap` of (variable, gradient). You can view
    var_grad as an ordered list of (key, (var, grad)) tuples. Every
    key of var_grad exists in vmap. Every variable in vmap that
    contributes to loss must exist in var_grad. Every var of var_grad
    must exist in vmap.  grad is the corresponding gradient computed
    for var. grad is guaranteed to be not None.
  """
  loss = HasRank(loss, 0)
  assert isinstance(vmap, NestedMap)

  # Uniqify and remove None.
  filtered_vmap = vmap.Filter(_Unique())
  assert filtered_vmap is not None

  # Filter out variables not contributing to 'loss'.
  trainable_variables = set(tf.trainable_variables())
  dependent_ops_and_tensors = set(FindNeeded([loss]))

  def Needed(v):
    if isinstance(v, tf.Variable):
      if v not in trainable_variables:
        # Skip non-trainable variables. Otherwise,
        # tf.Optimizer.apply_gradients throws up an exception instead
        # of skipping the update.
        return False
    # Not sure needed since tf.gradients will do this for us.
    return v.name in dependent_ops_and_tensors

  filtered_vmap = filtered_vmap.Filter(Needed)
  assert filtered_vmap is not None
  filtered_vlist = filtered_vmap.Flatten()

  # tpu vs non-tpu is slightly different.
  if use_tpu():
    if nas_run():
      take_grad = _ComputeGradientsTpuNas
    else:
      take_grad = _ComputeGradientsTpu
  else:
    take_grad = _ComputeGradientsSimple

  grads = take_grad(loss, filtered_vlist, grad_aggregation_method,
                    colocate_gradients_with_ops, gate_gradients)

  # Formulate pairs of (var, grad) and pack them into the same
  # structure as filtered_vmap.
  var_grad = filtered_vmap.Pack(list(zip(filtered_vlist, grads)))

  # Removes pairs whose grad is None.
  for key, (_, g) in var_grad.FlattenItems():
    if g is None:
      tf.logging.info('ComputeGradients drops %s', key)
  return var_grad.Filter(lambda v_g: v_g[1] is not None)


def MaskGradients(var_grad, grad_mask, grad_onehot):
  """Computes gradients of non-masked variables in vmap w.r.t loss.

  Args:
    var_grad: A `.NestedMap` of (variable, gradient)
    grad_mask: A `.NestedMap` of (variable, mask).
    grad_onehot: A 1-hot vector of the current data source selected.

  Returns:
    var_grad - a `.NestedMap` of (variable, mask * gradient).
  """

  def ApplyMask(entry):
    var, grad = entry
    grad_mask_dotproduct = tf.tensordot(grad_onehot, grad_mask[var.name], 1)
    if isinstance(grad, tf.IndexedSlices):
      return (var,
              tf.IndexedSlices(grad.values * grad_mask_dotproduct,
                               grad.indices))
    else:
      return (var, grad * grad_mask_dotproduct)

  return var_grad.Transform(ApplyMask)


def ApplyGradMultiplier(vs_gs_scale, grad_scale=None):
  """Scale gradients by grad_scale on same device as corresponding variables.

  Args:
    vs_gs_scale: A `.NestedMap` of (variable, gradient, scale).
    grad_scale: If None, each vs_gs entry has the scale. Otherwise, grad_scale
      applies to every entry.

  Returns:
    A `.NestedMap` of (variable, gradient * grad_scale). In particular, if
    grad_scale is 0, the result gradient is always 0, even if the input
    gradient is inf or nan.
  """

  if grad_scale is not None:
    vs_gs_scale = vs_gs_scale.Transform(lambda v_g: (v_g[0], v_g[1], grad_scale)
                                       )

  def ScaleOrZero(var, grad, scale):
    grad = CheckNumerics(grad, 'Gradient for %s is not finite.' % var.name)
    return tf.where(
        tf.equal(scale, 0.), tf.zeros_like(grad),
        tf.cast(scale, grad.dtype) * grad)

  def Scale(item):
    """Scales the gradient."""
    var, grad, scale = item
    assert grad is not None, ('No grad found for ', var.name)
    with tf.device(var.device):
      if isinstance(grad, tf.IndexedSlices):
        grad = tf.IndexedSlices(
            ScaleOrZero(var, grad.values, scale), grad.indices,
            grad.dense_shape)
      else:
        grad = ScaleOrZero(var, grad, scale)
    return (var, grad)

  return vs_gs_scale.Transform(Scale)


def ApplyGradNormCliping(vs_gs, norm=1.0):
  """Clip gradients to norm on same device as corresponding variables.

  Args:
    vs_gs: A `.NestedMap` of (variable, gradient).
    norm: Each tensor's gradient will be scaled down to have a maximum L2-norm
      value of `norm`.

  Returns:
    A `.NestedMap` of (variable, scaled_gradient). In particular, if
    grad_scale is 0, the result gradient is always 0, even if the input
    gradient is inf or nan.
  """
  vs_gs_norm = vs_gs.Transform(lambda v_g: (v_g[0], v_g[1], norm))

  def ClipByNorm(var, grad, norm):
    grad = CheckNumerics(grad, 'Gradient for %s is not finite.' % var.name)
    return tf.clip_by_norm(grad, norm)

  def Clip(item):
    """Scales the gradient."""
    var, grad, norm = item
    assert grad is not None, ('No grad found for ', var.name)
    with tf.device(var.device):
      if isinstance(grad, tf.IndexedSlices):
        grad = tf.IndexedSlices(
            ClipByNorm(var, grad.values, norm), grad.indices, grad.dense_shape)
      else:
        grad = ClipByNorm(var, grad, norm)
    return (var, grad)

  return vs_gs_norm.Transform(Clip)


SKIP_LP_REGULARIZATION = '__lingvo_skip_lp_regularization'


def AdjustGradientsWithLpLoss(var_grads, lp_regularizer_weight, p=2.0):
  """Adjusts the map of (var, grad) with Lp regularization, where p=1.0 or 2.0.

  Args:
    var_grads: a `.NestedMap` of (variable, gradient).
    lp_regularizer_weight: Lp regularization weight.
    p: For now we support 1.0 or 2.0.

  Returns:
    A tuple (lp_loss, var_grads).

    - lp_loss: A scalar. The lp loss.
    - var_grads: a `.NestedMap` of (variable, gradient) regulated by Lp.
  """
  # TODO(yuancao): For now we support p=1 or 2, but this can be extended to
  # lp-norm in general.

  assert p in [2.0, 1.0], 'For now we only support L1/L2 regularization.'

  def GetVar(item):
    var, grad = item
    if isinstance(grad, tf.IndexedSlices):
      with tf.device(var.device):
        ids = HasRank(grad.indices, 1)
        uniq_ids = tf.unique(ids).y
        return tf.gather(var, uniq_ids)
    else:
      return var

  def Skip(v_g):
    return v_g[0] not in tf.get_collection(SKIP_LP_REGULARIZATION)

  filtered_var_grads = var_grads.Filter(Skip)
  for k, (v, _) in filtered_var_grads.FlattenItems():
    tf.logging.info('AdjustGradientsWithLpLoss: %s: %s', k, v)

  if p == 2.0:
    lp_loss = 0.5 * lp_regularizer_weight * SumSquared(
        filtered_var_grads.Transform(GetVar).Flatten())
  elif p == 1.0:
    lp_loss = lp_regularizer_weight * SumAbs(
        filtered_var_grads.Transform(GetVar).Flatten())

  def LpGrad(item):
    """Adjusts item's grad w/ Lp loss term."""
    var, grad = item
    if isinstance(grad, tf.IndexedSlices):
      # Note: IndexedSlces appears for embedding lookups.
      # Embedding lookup ids can have duplicate. For duplicated ids, we
      # only want to consider once for each ids.
      with tf.device(var.device):
        emb = HasRank(var, 2)
        vocab_size = tf.shape(emb)[0]
        ids = HasRank(grad.indices, 1)
        values = tf.gather(emb, ids)  # [#ids, dims]
      with tf.device(grad.device):
        # Counts is a vector of size vocab_size. counts[i] is i-th words
        # occurances in 'ids'.
        counts = tf.unsorted_segment_sum(
            tf.ones_like(ids, dtype=values.dtype), ids, vocab_size)

        # Gradients for duplicated ids will be summed when they get
        # applied, and hence we account for that by first dividing
        # gradient resulting from lp loss by how many times the id is
        # duplicated.
        #
        # For each id in 'ids', we know counts[id] is non-zero,
        # hence, it's always safe to take reciprocal.
        weights = tf.reciprocal(tf.gather(counts, ids))
        weights = tf.expand_dims(weights, -1)  # [#ids, 1]
        if p == 2.0:
          grad_v = values
        elif p == 1.0:
          grad_v = tf.sign(values)
        delta = lp_regularizer_weight * weights * grad_v
        grad = tf.IndexedSlices(grad.values + delta, ids)
    elif var not in tf.get_collection(SKIP_LP_REGULARIZATION):
      with tf.device(var.device):
        if p == 2.0:
          grad_v = var
        elif p == 1.0:
          grad_v = tf.sign(var)
        delta = lp_regularizer_weight * grad_v
      with tf.device(grad.device):
        grad += delta
    return (var, grad)

  return lp_loss, var_grads.Transform(LpGrad)


def SplitRecursively(x, num_splits, axis=-1):
  """Splits Tensors in 'x' recursively.

  Args:
    x: a Tensor, or a list or NestMap containing Tensors to split.
    num_splits: number of splits per Tensor.
    axis: the split axis.

  Returns:
    A list of split values of length 'num_splits'.

    - If 'x' is a Tensor, a list of split Tensors.
    - If 'x' is a list, a list of lists, where each sublist has the same length
      as 'x' and the k'th element in each sublist corresponds to a split of the
      k'th element from 'x'.
    - If 'x' is a `.NestedMap`, a list of `.NestedMap`, where each field
      corresponds to a split from the same field of 'x'.
  """
  if isinstance(x, tf.Tensor):
    return tf.split(x, num_splits, axis=axis)
  elif isinstance(x, list):
    splits = [SplitRecursively(element, num_splits, axis) for element in x]
    splits = list(zip(*splits))
    return [list(t) for t in splits]
  elif isinstance(x, NestedMap):
    results = [NestedMap() for _ in range(num_splits)]
    for key, val in six.iteritems(x):
      val_splits = SplitRecursively(val, num_splits, axis)
      for i in range(num_splits):
        results[i][key] = val_splits[i]
    return results
  else:
    raise TypeError('Unexpected type for SplitRecursively: %s' % type(x))


def ConcatRecursively(splits, axis=-1):
  """Concatenates tensors from 'splits'.

  This is the inverse function of SplitRecursively.

  Args:
    splits: a list of splits to concatenate, where elements can be Tensors,
      lists, or `.NestedMap`. The elements must share the same type and
      structure.  For example, list elements must have the same length;
      `.NestedMap` must have the same set of fields.
    axis: the concatenation axis.

  Returns:
    Concatenated data.

    - If input 'splits' are Tensors, returns a concatenated Tensor.
    - If input 'splits' are lists, returns a list of the same length where the
      k'th element represents concatenated data of the k'th element from each
      split.
    - If input 'splits' are `.NestedMap`, returns a `.NestedMap` with each field
      concatenated from corresponding fields of input splits.

  Raises:
    TypeError: if 'splits' is not a list or elements of 'splits' do not have
      known or matching types.
    ValueError: if 'splits' is empty or elements of 'splits' do not have
      matching structures.
  """
  if not isinstance(splits, list):
    raise TypeError('Non-list inputs for ConcatRecursively: %s' % splits)
  if not splits:
    raise ValueError('Empty inputs for ConcatRecursively: %s' % splits)

  tmpl = splits[0]

  if isinstance(tmpl, tf.Tensor):
    return tf.concat(splits, axis=axis)
  elif isinstance(tmpl, list):
    if not all(isinstance(split, list) for split in splits):
      raise TypeError('Type mismatch for ConcatRecursively: %s' % splits)
    if not all(len(split) == len(tmpl) for split in splits):
      raise ValueError('Length mismatch for ConcatRecursively: %s' % splits)
    return [
        ConcatRecursively([split[i]
                           for split in splits], axis)
        for i in range(len(tmpl))
    ]
  elif isinstance(tmpl, NestedMap):
    if not all(isinstance(split, NestedMap) for split in splits):
      raise TypeError('Type mismatch for ConcatRecursively: %s' % splits)
    results = NestedMap()
    for key in tmpl:
      results[key] = ConcatRecursively([split[key] for split in splits], axis)
    return results
  else:
    raise TypeError('Unexpected type for ConcatRecursively: %s' % type(splits))


def AddToPruningCollections(weight, mask, threshold):
  """Add mask, threshold, and weight vars to their respective collections."""
  if mask not in tf.get_collection(pruning_layers.MASK_COLLECTION):
    tf.add_to_collection(pruning_layers.WEIGHT_COLLECTION, weight)
    tf.add_to_collection(pruning_layers.MASK_COLLECTION, mask)
    tf.add_to_collection(pruning_layers.THRESHOLD_COLLECTION, threshold)


def WeightedAvg(values, weights, sum_reduction_fn=tf.reduce_sum):
  """Computes weighted average of values from a tensor.

  Args:
    values: a tensor of values
    weights: a tensor of weights
    sum_reduction_fn: called to reduce the values and weights to single value.

  Returns:
    A tuple (avg, total_weight).

    - avg: weighted average value
    - total_weight: sum of all weights
  """
  values = with_dependencies([
      assert_equal(
          tf.shape(values),
          tf.shape(weights),
          message='shape of values and weights tensors must match.')
  ], values)
  total_weight = sum_reduction_fn(weights)
  avg = sum_reduction_fn(values * tf.cast(weights, values.dtype)) / tf.cast(
      total_weight, values.dtype)
  return avg, total_weight


def WeightedAvgOfMetrics(metrics):
  """Computes the weighted average of metrics in the list.

  Args:
    metrics: list of dictionaries of metrics

  Returns:
    ret_dict - dictionary of weighted averages of each metrics.
  """
  ret_dict = {}
  lists_of_metrics = {}
  for m in metrics:
    for name, (value, weight) in six.iteritems(m):
      if name not in lists_of_metrics:
        lists_of_metrics[name] = []
      lists_of_metrics[name].append((value, weight))

  for name, values_and_weights in sorted(six.iteritems(lists_of_metrics)):
    values = tf.stack([x[0] for x in values_and_weights])
    weights = tf.stack([x[1] for x in values_and_weights])
    ret_dict[name] = WeightedAvg(values, weights, tf.reduce_sum)

  return ret_dict


def ConcatPerExampleTensors(per_example):
  """Concatenate per-example tensors from many hosts into one large block.

  Args:
    per_example: list of dictionaries of per-example tensors.

  Returns:
    ret_dict - string -> concatenated tensors.
  """
  ret_dict = {}
  lists_of_per_example = {}
  for m in per_example:
    for name, value in six.iteritems(m):
      if name not in lists_of_per_example:
        lists_of_per_example[name] = []
      lists_of_per_example[name].append(value)

  for name, values in sorted(six.iteritems(lists_of_per_example)):
    ret_dict[name] = tf.concat(values, 0)

  return ret_dict


def CombineMetrics(loss_metric_weight_pairs):
  """Combines metrics from `loss_metric_weight_pairs` according to weights.

  Keys must either exist in all metrics, in which it will be processed as a
  weighted sum, or exist in only one metrics, in which case it will be copied.

  Args:
    loss_metric_weight_pairs: a list of (metrics, weight) pairs, where each
      weight is a float and each metrics is a dict with str keys and
      (metric_value, target_weight) values.

  Returns:
    A dict with the same set of keys as input metrics and values of
    (weighted_sum(metric_value), weighted_sum(target_weight)).

  Raises:
    ValueError: if there exists a metric that exists in more than one element
      of `loss_metric_weight_pairs` but not in all of them.
  """
  all_keys = set([
      k for loss_metrics, _ in loss_metric_weight_pairs
      for k in six.iterkeys(loss_metrics)
  ])
  result = {}
  for k in all_keys:
    count = 0
    for loss_metrics, weight in loss_metric_weight_pairs:
      if k in loss_metrics:
        count += 1
    if count > 1 and count != len(loss_metric_weight_pairs):
      raise ValueError('Found metric %s which exists in more than one'
                       'but not all loss metrics.' % k)

    total_val = 0
    total_target_weight = 0
    for loss_metrics, weight in loss_metric_weight_pairs:
      if k in loss_metrics:
        val, target_weight = loss_metrics[k]
        if count == 1:
          # Single metric, don't multiply by weight.
          total_val = val * target_weight
          total_target_weight = target_weight
        else:
          # Total weighted sum of all predictions.
          total_val += weight * val * target_weight
          total_target_weight += weight * target_weight

    result[k] = (total_val / total_target_weight, total_target_weight)
  return result


def _AddVN(p, x, step=None):
  assert p.vn.scale is not None
  seed = p.vn.seed
  if seed and step:
    seed += step * 203984
  noises = tf.cast(p.vn.scale, x.dtype) * tf.random_normal(
      tf.shape(x), stddev=1.0, seed=seed, dtype=x.dtype)
  return x + noises


def AddGlobalVN(params, weights):
  """Adds variational noise to weights if specified by params."""
  p = params
  if p.vn.global_vn:
    weights = _AddVN(p, weights)
  return weights


def AddPerStepVN(params, weights, step=None):
  """Adds per-setp variational noise to weights if specified by params."""
  p = params
  if p.vn.per_step_vn:
    weights = _AddVN(p, weights, step)
  return weights


def VariationalNoiseParams(scale, global_vn=False, per_step_vn=False,
                           seed=None):
  """Returns a hyperparams for variational noise."""
  p = hyperparams.Params()
  p.Define(
      'scale', scale,
      'Std of the variational noise to apply . This can be a scalar,'
      ' or a scalar tensor.')
  p.Define('global_vn', global_vn,
           'Adds global variational noise every training setp iff True.')
  p.Define('per_step_vn', per_step_vn,
           'Adds per-timesetp variational noise iff True.')
  p.Define('seed', seed, 'Random seed used to generate noise.')
  return p


def GetStepSeed():
  """Gets step_seed."""
  step_seed_tensors = tf.get_default_graph().get_collection_ref('step_seed')
  if not step_seed_tensors:
    ResetStepSeed()
    return GetStepSeed()
  elif len(step_seed_tensors) == 1:
    return step_seed_tensors[0]
  else:
    raise ValueError('Multiple tensors in step_seed collection.')


def ResetStepSeed(seed=0):
  """Resets step_seed to specified value."""
  new_step_seed = tf.convert_to_tensor(seed, dtype=tf.int64)
  step_seed_tensors = tf.get_default_graph().get_collection_ref('step_seed')
  if len(step_seed_tensors) == 1:
    step_seed_tensors[0] = new_step_seed
  elif not step_seed_tensors:
    tf.add_to_collection('step_seed', new_step_seed)
  else:
    raise ValueError('Multiple tensors in step_seed collection.')


def GetIncStepSeed():
  """Returns and increments the step_seed."""
  step_seed = GetStepSeed()
  # TODO(lepikhin): introduce a routine filling a queue of uint32 random seeds
  # independent of underlying PRNG used by tensorflow.
  ResetStepSeed(step_seed + 1)
  return step_seed


def GenerateStepSeedPair(p, global_step, op_seed=None):
  """Generates a seed pair for deterministic random operations in functional loops.

  This function retrieves a unique seed pair on each call, based off the current
  global step and step seed. The step seed ensures this function returns a
  unique seed pair on each call: calling this function automatically increments
  the step seed. The step seed is automatically reset at the beginning of each
  global step in the model's FProp and works transparently through recurrent.py.

  Args:
    p: A hyperparams.Params object, containing keys 'random_seed' and
      'is_inference'.
    global_step: The global step.
    op_seed: An additional operation-level seed to apply.

  Returns:
    A size 2 tensor of op seeds to use for stateless_random ops.
  """
  seed_dtype = tf.int32 if use_tpu() else tf.int64
  if p.is_inference and p.random_seed is None:
    # Ensure GetIncStepSeed is called even inside the shortcut.
    # This ensures if p.random_seed is set for other ops that use this function
    # that they will get the same seed pair whether or not p.random_seed is set
    # for this specific call.
    GetIncStepSeed()
    # Unlike tf.random*, stateless random ops are completely determined by the
    # passed-in seeds. This means at inference time the same inputs will produce
    # the same outputs, even if the model is supposed to have randomness such as
    # dropout during inference. We inject additional randomness only during
    # inference if the graph is exported with random_seed=None as a workaround.
    return tf.random_uniform([2], maxval=seed_dtype.max, dtype=seed_dtype)

  global_step = tf.cast(global_step, seed_dtype)
  step_seed = tf.cast(GetIncStepSeed(), seed_dtype)
  seeds = tf.stack([global_step, step_seed])

  if p.random_seed is not None:
    seeds += p.random_seed
  if op_seed is not None:
    seeds += op_seed
  return seeds


def DeterministicDropout(x, keep_prob, seeds, noise_shape=None, name=None):
  """Similar to `tf.nn.dropout()`, but fully deterministic.

  Args:
    x: A float Tensor on which to apply dropout.
    keep_prob: A scalar `Tensor` of keep probability.
    seeds: A Tensor of shape [2]. 2 seeds for deterministic random number
      generator.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the shape for
      randomly generated keep/drop flags.
    name: An optional name for this operation.

  Returns:
    A Tensor with the same shape as `x`.

  Raises:
    InvalidArgumentError: if keep_prob is invalid.
  """
  if isinstance(keep_prob, numbers.Real):
    if keep_prob <= 0 or keep_prob > 1:
      raise tf.errors.InvalidArgumentError(
          'keep_prob must be in range (0, 1]. Value: {}'.format(keep_prob))

    if keep_prob == 1:
      return x
  with tf.name_scope(name, 'dropout', [x]) as name:
    if use_tpu():
      seeds = tf.cast(seeds, tf.int32)
    keep_prob = tf.convert_to_tensor(
        keep_prob, dtype=tf.float32, name='keep_prob')
    # uniform in [keep_prob, 1.0 + keep_prob)
    # StatelessRandomUniform op does not support non-float (e.g. bfloat16) dtype
    # and non-int32 seed types.
    noise_shape = noise_shape or GetShape(x)
    random_tensor = keep_prob + tf.contrib.stateless.stateless_random_uniform(
        noise_shape, seed=seeds, dtype=tf.float32)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    if x.dtype != tf.float32:
      binary_tensor = tf.cast(binary_tensor, x.dtype)
      keep_prob = tf.cast(keep_prob, dtype=x.dtype)
    result = tf.div(x, keep_prob) * binary_tensor
    result.set_shape(x.get_shape())
    return result


BATCH_NORM_UPDATES = 'batch_norm_updates'

_BATCH_NORM_UPDATES_DICT = '__batch_norm_update_dict'
_get_batch_norm_updates_dict = _CollectionGetter(
    _BATCH_NORM_UPDATES_DICT, lambda: {})


def UpdateBatchNormVars(batch_norm_var, batch_norm_stats, decay):
  """Update batch normalization moving averages."""
  with tf.name_scope(
      'AssignMovingAvg', values=[
          batch_norm_var,
          batch_norm_stats,
          decay,
      ]) as scope:
    with tf.colocate_with(batch_norm_var):
      decay = tf.convert_to_tensor(
          1.0 - decay, dtype=batch_norm_var.dtype.base_dtype)
      update_delta = (batch_norm_var - batch_norm_stats) * decay
      bn_update = tf.assign_sub(batch_norm_var, update_delta, name=scope)
  tf.add_to_collection(BATCH_NORM_UPDATES, bn_update)
  bn_update_dict = _get_batch_norm_updates_dict()
  assert bn_update.name not in bn_update_dict
  bn_update_dict[bn_update.name] = (batch_norm_var, batch_norm_stats)
  return bn_update


def FindRelevantBatchNormUpdates(loss, batch_norm_updates):
  """Finds and returns a list of relevant batch-normalization updates.

  Args:
    loss: The loss that is being optimized for.
    batch_norm_updates: A list of batch normalization updates.

  Returns:
    A pair of lists. The first list contains all the batch normalization updates
    that are relevant to the loss being optimized, and the second list contains
    all in batch_norm_updates but not in the first list.
  """
  dependent_ops_and_tensors = set(FindNeeded([loss]))
  relevant_updates = []
  irrelevant_updates = []

  bn_update_dict = _get_batch_norm_updates_dict()
  for bn_update in batch_norm_updates:
    assert bn_update.name in bn_update_dict, (
        '%s is probably not a valid batch normalization update op.'
        ' Make sure batch normalization is done through calling'
        ' the py_utils.UpdateBatchNormVars helper routine.')
    bn_stat_name = bn_update_dict[bn_update.name][1].name
    if bn_stat_name in dependent_ops_and_tensors:
      # If a batch normalization stat is computed in the forward pass in
      # computing loss, then the corresponding batch normalization update is
      # relevant. Otherwise, it is not.
      relevant_updates.append(bn_update)
    else:
      irrelevant_updates.append(bn_update)
  return relevant_updates, irrelevant_updates


_MODEL_SPLIT_ID_STACK = '__model_split_id_stack'
_get_model_split_id_stack = _CollectionGetter(
    _MODEL_SPLIT_ID_STACK, lambda: [0])


def GetModelSplit():
  return _get_model_split_id_stack()[-1]


@contextlib.contextmanager
def ModelSplit(split_id):
  assert split_id >= 0
  _get_model_split_id_stack().append(split_id)
  yield
  _get_model_split_id_stack().pop()


_SAMPLE_STEP_KEY = 'sample_step'


@contextlib.contextmanager
def SampleStep(step):
  """A context for a sample step during decoding.

  Example usage::

      with py_utils.SampleStep(step):
        sample = self.DecodeOneStep()

  Args:
    step: the step tensor.

  Yields:
    a context manager for the step scope.
  """
  stack = tf.get_collection_ref(_SAMPLE_STEP_KEY)
  try:
    stack.append(step)
    yield step
  finally:
    stack.pop()


def _GetSampleStep():
  stack = tf.get_collection(_SAMPLE_STEP_KEY)
  return stack[-1] if stack else None


def AddDebugTensor(tensor, summarize=None, name=None):
  """Adds `tensor` to the debug collection.

  Prints the tensor if `--print_debug_tensors` is True.

  Args:
    tensor: A tensor.
    summarize: Only print this many entries of each tensor. If None, then a
      maximum of 3 elements are printed per input tensor.
    name: An optional name for the tensor.

  Returns:
    A Tensor that evaluates to the same value as the input tensor.
  """
  if FLAGS.print_debug_tensors:
    step = _GetSampleStep()
    tensors_to_print = ([] if step is None else [step]) + [tensor]
    with tf.name_scope(name) as s:
      tensor = tf.Print(
          tensor,
          tensors_to_print,
          message='DEBUG tensor %s' % s,
          name=name,
          summarize=summarize)
  return tensor


def ArgMax(inputs):
  """tf.argmax wrapper.

  Args:
    inputs: A tensor, whose last dimension is being reduced on.

  Returns:
    A tensor of rank tf.rank(logits)-1. If i == ret[indices],
    logits[indices, i] is the maximum among logits[indices, :].
  """
  if use_tpu():
    return tf.argmax(inputs, axis=-1, output_type=tf.int32)
  else:
    return tf.argmax(inputs, axis=-1)


def _EnsureMatrixShape(x):
  if x.shape.ndims is None:
    x.set_shape([None, None])
  else:
    assert x.shape.ndims == 2
  return x


def Matmul(x, y, *args, **kwargs):
  """tf.matmul wrapper expecting x and y are actually matrices."""
  x = _EnsureMatrixShape(x)
  y = _EnsureMatrixShape(y)
  return tf.matmul(x, y, *args, **kwargs)


def clip_by_value(t, clip_value_min, clip_value_max, name=None):
  if t.dtype.is_complex:
    return tf.complex(
        tf.clip_by_value(
            tf.real(t), clip_value_min, clip_value_max, '%s_real' % name),
        tf.clip_by_value(
            tf.imag(t), clip_value_min, clip_value_max, '%s_imag' % name))
  return tf.clip_by_value(t, clip_value_min, clip_value_max, name)


def _TransformAndSum(tensor_list, transform):
  with tf.name_scope('TransformAndSum'):
    sum_transform = []
    for t in tensor_list:
      with tf.device(t.device):
        if isinstance(t, tf.IndexedSlices):
          sum_transform += [tf.reduce_sum(transform(t.values))]
        else:
          sum_transform += [tf.reduce_sum(transform(t))]
    return tf.add_n(sum_transform)


def SumSquared(tensor_list):
  return _TransformAndSum(tensor_list, lambda v: tf.abs(v)**2)


def SumAbs(tensor_list):
  return _TransformAndSum(tensor_list, tf.abs)


def PiecewiseConstant(x_in, boundaries, values, vdtype):
  """Returns the piecewise value of x_in."""
  x_in = tf.cast(tf.convert_to_tensor(x_in), tf.float32)
  assert len(values) == len(boundaries) + 1
  assert sorted(boundaries) == list(boundaries)
  bs = tf.convert_to_tensor(boundaries, dtype=tf.float32)
  vs = tf.convert_to_tensor(values, dtype=vdtype)
  # The following is equivalent to 'return vs[index]'.
  index = tf.reduce_sum(tf.cast(tf.greater(x_in, bs), tf.int32))
  one_hot_vec = tf.one_hot(
      tf.expand_dims(index, 0), depth=len(values), dtype=vdtype)
  return Matmul(tf.reshape(vs, (1, -1)), tf.transpose(one_hot_vec))[0][0]


def PadSequenceDimension(x, length, pad_val, shape=None):
  """Pads x to `length` using `pad_val` along the second dim.

  Assumes `x` is a tensor with rank >= 2, and it only pads `x` to `length`
  along the second dim. Explicitly sets the returned tensor shape to `shape` if
  given. Raises runtime errors if x.shape[1] > length or x.shape[i] != shape[i]
  where i != 1.

  Args:
    x: the tensor to be padded with shape [batch, seq_len, ...].
    length: an int to specify the length to pad x to.
    pad_val: an int or float used to pad x.
    shape: an int array specifying the shape of the padded tensor if specified.

  Returns:
    The padded tensor with shape [batch, seq_len, ...], where
    ret[:, :seq_len, ...] == x.
  """
  if x.shape.ndims is not None:
    rank = x.shape.ndims
    assert rank >= 2
    slen = GetShape(x, rank)[1]
    pad_len = length - slen
    pad = [[0, 0] for _ in range(rank)]
    pad[1][1] = pad_len
  else:
    rank = tf.rank(x)
    with tf.control_dependencies([assert_greater_equal(rank, 2)]):
      slen = tf.shape(x)[1]
    pad_len = length - slen
    pad = tf.scatter_nd([[1, 1]], [pad_len], [rank, 2])
  x = tf.pad(x, pad, constant_values=pad_val)
  if x.shape.ndims is not None and isinstance(length, int):
    static_shape = x.shape.as_list()
    static_shape[1] = length
    x.set_shape(static_shape)

  if shape:
    if not isinstance(shape, (list, tuple)):
      raise TypeError('Shape must be a list or tuple.')
    x = HasRank(x, len(shape))
    x = tf.ensure_shape(x, shape)
  return x


def ApplyPadding(padding, x, padded=None, broadcast=True, use_select=True):
  """Applies padding to a tensor.

  This is preferable to using arithmetic means for masking out padded values
  such as::

      # Equiv to ApplyPadding(padding, x))
      x *= 1.0 - padding
      # Equiv to ApplyPadding(padding, new, old)
      new = old * padding + new * (1 - padding)

  Aside from just being easier to read and reason about, using this function
  is friendly to quantized representations because it does not mix arithmetic
  on the padding values with the values in the tensor being padded (which can
  have a very different range than the 0..1 padding tensor).

  In addition, this works around issues in quantized schemes where we are
  guaranteed to have an exact 0 but not necessarily any other number (i.e. 1).

  Args:
    padding: Tensor of padding values where 0 == keep and 1 == pad.
    x: Tensor to apply padding to.
    padded: Optional. Values to include for padded elements. Defaults to zeros.
      Must be the same shape as 'x' if specified.
    broadcast: Whether to broadcast the padding shape to the shape of 'x'. You
      almost certainly want this to be true as it matches how padding would be
      expanded if applied arithmetically.
    use_select: Controls whether padding is applied with a select-mask
      (True/default) or arithmetically (False). Some platforms have a
      sensitivity to one or the other and this is used to work around such
      issues.

  Returns:
    A tensor with the same shape as x with padded values masked.
  """
  padding = with_dependencies([
      Assert(
          tf.reduce_all(
              tf.logical_or(tf.equal(padding, 0.0), tf.equal(padding, 1.0))),
          [padding])
  ], padding)
  if use_select:
    if padded is None:
      padded = tf.zeros_like(x)
    if broadcast:
      padding *= tf.ones_like(x)  # Broadcast padding to the full shape.
    return tf.where(padding > 0.0, padded, x)
  else:
    if padded is None:
      return x * (1.0 - padding)
    else:
      return x * (1.0 - padding) + padded * padding


def TrimTrailingPaddings(inputs, paddings):
  """Trims trailing paddings from inputs.

  Since the number of dimensions is not fixed, this will not work on TPU.

  Args:
    inputs: a tensor with shape [batch, length, ...].
    paddings: a tensor with shape [batch, length].

  Returns:
    Trimmed inputs and paddings. For compatibility reasons, the trimmed tensors
    will always have length at least 1.
  """
  paddings = HasRank(paddings, 2)
  # Find the last unpadded value. Argmax returns the first index when tied.
  # Cannot just use tf.reduce_sum because there might be leading paddings.
  cumsum = tf.cumsum(1.0 - paddings, axis=1)
  length = tf.argmax(cumsum, axis=1, output_type=tf.int32) + 1
  max_length = tf.reduce_max(length)
  output_shape = tf.shape(inputs)
  output_shape = tf.concat([[output_shape[0], max_length], output_shape[2:]],
                           axis=0)
  outputs = tf.slice(inputs, tf.zeros_like(output_shape), output_shape)
  out_paddings = tf.slice(paddings, [0, 0],
                          tf.stack([output_shape[0], max_length]))
  return outputs, out_paddings


def ReversePaddedSequence(inputs, paddings):
  """Reverse inputs based on paddings.

  Only reverse the unpadded portion of `inputs`. It assumes inputs are only
  padded in the end.

  Args:
    inputs: a tensor of [seq_length, batch_size, num_input_nodes].
    paddings: a tensor of float32/float64 zero or one of shape [seq_length,
      batch_size, 1].

  Returns:
    A reversed tensor of the same shape as `inputs`.
  """
  inversed_paddings = 1.0 - tf.squeeze(paddings, 2)
  inputs_length = tf.cast(
      tf.rint(tf.reduce_sum(inversed_paddings, axis=0)), dtype=tf.int32)
  return tf.reverse_sequence(inputs, inputs_length, seq_axis=0, batch_axis=1)


def Retry(*args, **kwargs):
  return retry.Retry(*args, **kwargs)


# FailedPreconditionError: variables are not initialized.
# AbortedError: processes restarts.
# UnavailableError: Bad hardware status: 0x1
transient_tf_errors = (tf.errors.FailedPreconditionError,
                       tf.errors.AbortedError, tf.errors.UnavailableError)


def RetryOnTransientTfError(*args, **kwargs):
  return Retry(transient_tf_errors, *args, **kwargs)


def PadOrTrimTo(x, shape, pad_val=0):
  """Pad and slice x to the given shape.

  Args:
    x: A tensor.
    shape: The shape of the returned tensor.
    pad_val: An int or float used to pad x.

  Returns:
    'x' is padded with pad_val and sliced so that the result has the given
    shape.
  """
  if isinstance(shape, (list, tuple)):
    expected_rank = len(shape)
  elif isinstance(shape, tf.TensorShape):
    expected_rank = shape.rank
  else:
    shape = HasRank(shape, 1)
    expected_rank = tf.size(shape)
  x = HasRank(x, expected_rank)
  # If dim-i is less than shape[i], pads on the right shape[i] -
  # dim-i.  Otherwise, pads [0, 0] for dim-i.
  pad = shape - tf.minimum(tf.shape(x), shape)
  zeros = tf.zeros_like(pad)
  x = tf.pad(x, tf.stack([zeros, pad], axis=1), constant_values=pad_val)
  # If dim-i is larger than shape[i], we slice [0:shape[i]] for dim-i.
  return tf.reshape(tf.slice(x, zeros, shape), shape)


def RepeatDim(tensor, multiple, axis):
  """Copies elements in tensor's axis "multiple" times, like np.repeat."""
  # x = [[1, 2, 3], [4, 5, 6]]
  # RepeatDim(x, multiple=2, axis=1) gives:
  # [[1, 1, 2, 2, 3, 3]. [4, 4, 5, 5, 6, 6]]
  # As a comparison tf.tile(x, multiples=[1, 2]) gives:\
  # [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]]

  if multiple == 1:
    return tensor
  t_shape = tf.shape(tensor)
  tensor_dims = tf.concat(
      [t_shape[:axis], [t_shape[axis] * multiple], t_shape[axis + 1:]], 0)
  multiple_dims = tf.concat([
      tf.fill([axis + 1], 1), [multiple],
      tf.fill([tf.rank(tensor) - axis - 1], 1)
  ], 0)
  return tf.reshape(
      tf.tile(tf.expand_dims(tensor, axis + 1), multiple_dims), tensor_dims)


def StackTensorsRecursively(values):
  """Recursively stacks Tensors in a list of `.NestedMap`.

  Args:
    values: a list of `.NestedMap` or Tensors to stacks.

  Returns:
    A `.NestedMap` with stacked values or a stacked Tensor.
  """
  flatten = [w.Flatten() for w in values]
  stacked = []
  for i in range(len(flatten[0])):
    stacked += [tf.stack([flatten[j][i] for j in range(len(flatten))])]
  ret = values[0].Pack(stacked)
  return ret


def MixByWeight(inputs, weights):
  """Returns a weighted random choice and bprop type from the give inputs.

  Args:
    inputs: a list of callables, where each callable returns a tf.Tensor or a
      nested structure containing tf.Tensor. Function return types must be
      consistent across elements. The tf.Operation to compute the result tensor
      will only be invoked for one input at a time. For example, if each fn
      represents an input record stream, a record will be drawn only from a
      selected stream while the other streams will remain unchanged.
    weights: a 1D tensor of float > 0 of the same length as inputs.

  Returns:
    A probablistic sample from the inputs proportional to the weights. The
    return type will be the same as return type of individual 'fn' from the
    inputs.
    A one-hot vector of the source selected.
  """
  weights = tf.convert_to_tensor(weights, dtype=tf.float32)
  weights = with_dependencies([
      assert_equal(tf.shape(weights), [len(inputs)]),
      assert_greater_equal(tf.reduce_min(weights), 0.0)
  ], weights)

  lower = tf.cumsum(weights, exclusive=True)
  upper = tf.cumsum(weights, exclusive=False)
  r = tf.random_uniform(shape=[], maxval=upper[-1])
  return_input = tf.case(
      [(tf.logical_and(lower[i] <= r, r < upper[i]), inputs[i])
       for i in range(len(inputs))],
      exclusive=True)
  selected_index = tf.case(
      [(tf.logical_and(lower[i] <= r, r < upper[i]), lambda i=i: i)
       for i in range(len(inputs))],
      exclusive=True)
  bprop_index = tf.one_hot(selected_index, len(inputs), dtype=tf.float32)
  return return_input, bprop_index


def CheckShapes(shapes):
  """Asserts that shapes is a tuple of fully defined tf.TensorShape."""
  assert isinstance(shapes, tuple), str(shapes)
  for s in shapes:
    assert isinstance(s, tf.TensorShape), str(s)
    assert s.is_fully_defined()


def FPropDtype(params):
  return params.fprop_dtype if params.fprop_dtype is not None else params.dtype


def UpdateFpropDtype(params, fprop_dtype):
  """Recursively update the fprop_dtype of the Params."""
  for key, val in params.IterParams():
    if isinstance(val, hyperparams.Params):
      UpdateFpropDtype(val, fprop_dtype)
    elif key == 'fprop_dtype':
      params.fprop_dtype = fprop_dtype


def UpdateDtype(params, dtype):
  """Recursively update the dtype of the Params."""
  for key, val in params.IterParams():
    if isinstance(val, hyperparams.Params):
      UpdateDtype(val, dtype)
    elif key == 'dtype':
      params.dtype = dtype


def NameScopeDecorator(name_scope):
  """Decorates a python function to introduce a tf.name_scope.

  Example::

      @py_utils.NameScopeDecorator('foobar')
      def MyFoobarMethod(self):
        # ... Do TF things

  Args:
    name_scope: The name scope to introduce.

  Returns:
    A function decorator.
  """

  def Decorator(f):

    def Wrapped(*args, **kwargs):
      with tf.name_scope(name_scope):
        return f(*args, **kwargs)

    return Wrapped

  return Decorator


def SequencesToDebugStrings(ids, lens, summarize=5):
  """Returns debug strings for the given sequences.

  Args:
    ids: int32 of [batch, len].
    lens: int32 of [batch].
    summarize: number of ids to summarize per sequence.

  Returns:
    A string tensor of [batch].
  """
  num_seqs = tf.shape(lens)[0]

  def _Body(i, result):
    line = tf.strings.format('{}', ids[i, :lens[i]], summarize=summarize)
    return i + 1, tf.concat([result, tf.reshape(line, [1])], axis=0)

  i0 = tf.zeros(shape=[], dtype=tf.int32)
  result0 = tf.constant('', shape=[0], dtype=tf.string)
  _, strs = tf.while_loop(
      lambda i, result: i < num_seqs,
      _Body, (i0, result0),
      shape_invariants=(i0.shape, tf.TensorShape([None])))
  return strs


def RematerializeFn(fn, *xs):
  """Calls fn and rematerializes fn in the backward pass.

  fn(*xs) -> ys, where xs and ys can be a single tensor or a tuple of tensors.

  Args:
    fn: A python function to be rematerialized in the backprop pass.
    *xs: A single tensor or a list/tuple of tensors. 'xs' are input args to the
         fn function.
  Returns:
    fn(*xs)
  """

  def Backward(op, *dy):
    """The backward function that rematerializes forward outputs."""
    always_true = tf.random.uniform([]) < 2.0
    # Alternatively, can do this:
    # tf.where(tf.is_nan(x),
    #          tf.constant(float('nan'), dtype=x.dtype) * tf.ones_like(x),
    #          x)
    xs = [
        tf.where(always_true, x, tf.zeros(GetShape(x), dtype=x.dtype))
        for x in op.inputs
    ]
    ys = fn(*xs)
    dxs = tf.gradients(ys, xs, grad_ys=dy)
    dxs_final = []
    for dx, x in zip(dxs, xs):
      if dx is None:
        dxs_final.append(tf.zeros_like(x))
      else:
        dxs_final.append(dx)
    assert len(dxs_final) == len(xs)
    return tuple(dxs_final)

  xs_dtypes = [x.dtype for x in xs]

  @function.Defun(*xs_dtypes, python_grad_func=Backward)
  def Forward(*xs):
    """Forward function plus sanity checks."""
    ys = fn(*xs)
    # Some sanity check.
    assert not function.get_extra_inputs()
    assert not function.get_extra_args()
    assert not function.get_extra_vars()
    if isinstance(ys, tuple):
      for y in ys:
        assert isinstance(y, tf.Tensor)
    else:
      assert isinstance(ys, tf.Tensor)

    return ys

  return Forward(*xs)
