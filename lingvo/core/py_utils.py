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
"""Common utilities."""

# ==============================================================================
# Note: Avoid adding dependencies to py_utils beyond standard python packages
#       and tensorflow.
# ==============================================================================

import collections as py_collections
import contextlib
import functools
import hashlib
import inspect
import math
import numbers
import os
import pkgutil
import re
import threading
import traceback

import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import gshard_utils
from lingvo.core import hyperparams
from lingvo.core import nested_map
from lingvo.core import ops
from lingvo.core import py_utils_flags
from lingvo.core import retry
from lingvo.core import symbolic
from lingvo.core import thread_local_utils
from lingvo.core import tshape

import numpy as np
import six

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.tf2 import enabled as tf2_enabled
from tensorflow.python.tpu import topology as tf_topology
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util import deprecation
# pylint: enable=g-direct-tensorflow-import

FLAGS = tf.flags.FLAGS

# pylint: disable=protected-access
_FromGlobal = py_utils_flags._FromGlobal
# pylint: enable=protected-access
use_xla = py_utils_flags.use_xla
use_tpu = py_utils_flags.use_tpu
testonly_skip_norm_layers = py_utils_flags.testonly_skip_norm_layers
tpu_compat = py_utils_flags.tpu_compat
use_stateless_vars_init = py_utils_flags.use_stateless_vars_init

ENQUEUE_OPS = '__lingvo_enqueue_ops'

# pylint: disable=protected-access
deprecation._PRINT_DEPRECATION_WARNINGS = False

# pylint: enable=protected-access

ThreadLocalStack = thread_local_utils.ThreadLocalStack
ThreadLocalDict = thread_local_utils.ThreadLocalDict
NestedMap = nested_map.NestedMap


def Assert(condition, data, *args, **kwargs):
  if py_utils_flags.enable_asserts():
    return tf.Assert(condition, data, *args, **kwargs)
  else:
    return tf.no_op()


def assert_equal(*args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    return tf.assert_equal(*args, **kwargs)
  else:
    return tf.no_op()


def assert_greater_equal(*args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    return tf.debugging.assert_greater_equal(*args, **kwargs)
  else:
    return tf.no_op()


def assert_greater(*args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    return tf.assert_greater(*args, **kwargs)
  else:
    return tf.no_op()


def assert_less_equal(*args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    return tf.debugging.assert_less_equal(*args, **kwargs)
  else:
    return tf.no_op()


def assert_less(*args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    return tf.assert_less(*args, **kwargs)
  else:
    return tf.no_op()


def assert_between(x, l, r, *args, **kwargs):  # pylint: disable=invalid-name
  x = tf.convert_to_tensor(x)
  l = tf.cast(tf.convert_to_tensor(l), x.dtype)
  r = tf.cast(tf.convert_to_tensor(r), x.dtype)
  return tf.group([
      assert_greater_equal(x, l, *args, **kwargs),
      assert_less(x, r, *args, **kwargs)
  ])


def assert_shape_match(*args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    filepath, line, func, _ = traceback.extract_stack(limit=3)[-2]
    kwargs['msg'] = 'LINGVO ASSERT %s:%s(%s)' % (re.sub(
        r'.*/', '', filepath), line, func)
    return ops.assert_shape_match(*args, **kwargs)
  else:
    return tf.no_op()


def assert_same_dim0(xs, *args, **kwargs):  # pylint: disable=invalid-name
  if py_utils_flags.enable_asserts():
    return ops.assert_same_dim0(xs, *args, **kwargs)
  else:
    return tf.no_op()


def assert_even_divide(denorm, num):  # pylint: disable=invalid-name
  """Asserts that denorm is evenly divided by num."""
  denorm = tf.convert_to_tensor(denorm)
  num = tf.convert_to_tensor(num)

  if denorm.dtype not in (tf.int32, tf.int64):
    raise ValueError('denorminator.dtype is not tf.int32 or tf.int64.')
  if num.dtype not in (tf.int32, tf.int64):
    raise ValueError('numerator.dtype is not tf.int32 or tf.int64.')

  num = HasShape(num, GetShape(denorm))

  quo = denorm // num
  return assert_equal(quo * num, denorm)


def AssertIdShape(expected_ids_shape_pattern, ids_shape, *args):
  """Asserts shape expected_ids_shape_pattern matches all other input shapes."""

  def AssertFn(inputs):
    dependencies = [
        assert_shape_match(inputs.ids_shape, inputs.expected_ids_shape_pattern)
    ] + [
        assert_shape_match(inputs.ids_shape, x_shape) for x_shape in inputs.args
    ]
    return with_dependencies(dependencies, inputs.ids_shape)

  inputs = NestedMap(
      expected_ids_shape_pattern=expected_ids_shape_pattern,
      ids_shape=ids_shape,
      args=args)
  return CallDefun(AssertFn, Transform(tf.convert_to_tensor, inputs))


def _CheckNumerics(x, message=None, *args, **kwargs):
  if x.dtype.is_floating:
    x_name = x.name if not tf.executing_eagerly() else '[eager]'
    if 'name' not in kwargs:
      kwargs['name'] = re.sub(r':\d+', '', x_name) + '_CheckNumerics'
    return tf.debugging.check_numerics(x, message if message else x_name, *args,
                                       **kwargs)
  else:
    return x


def CheckNumerics(inp, message=None, *args, **kwargs):
  """Check numerics for tensors in inp."""
  if not py_utils_flags.enable_check_numerics():
    return inp
  if isinstance(inp, list):
    return [_CheckNumerics(x, message, *args, **kwargs) for x in inp]
  if isinstance(inp, tuple):
    return tuple(_CheckNumerics(x, message, *args, **kwargs) for x in inp)
  return _CheckNumerics(inp, message, *args, **kwargs)


def with_dependencies(dependencies, output_tensor):  # pylint: disable=invalid-name
  with tf.control_dependencies(dependencies):
    return tf.identity(output_tensor)


@contextlib.contextmanager
def _PrintOptions(*args, **kwargs):
  original = np.get_printoptions()
  np.set_printoptions(*args, **kwargs)
  try:
    yield
  finally:
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


def Debug(tensor, message='', enabled=True, summarize=100, more=None):
  """Wrapper around tf.Print() and tf.logging.info() to simplify debug printing.

  x = py_utils.Debug(x)

  When the graph is built a regular log info line will be printed:
  -DBG- py_utils_test.py:429 x=Tensor(...

  Then when the tensor node is evaluated it will print lines like:
  -DBG- py_utils_test.py:429 x Const:0[x.shape=][2 2][x=][[1 2][3 4]]

  WARNING: The code that parses local variable names can fail. E.g. don't write
  two Debug() calls on one line or a Debug() call that spans more than one line.

  Args:
     tensor: A tensor to print.
     message: A message to print.
     enabled: To enable the debugging.
     summarize: Integer with number of tensor values to print.
     more: An optional list of additional tensors.

  Returns:
    The tensor.
  """
  if not enabled or _FromGlobal('disable_py_utils_debug'):
    return tensor

  if more is None:
    more = []

  stack = inspect.stack()[1][0]
  caller = inspect.getframeinfo(stack)

  caller_var = ''
  caller_more_vars = []
  if caller.code_context:
    # Rough and likely to fail. But better than nothing.
    caller_var = re.compile(r'Debug\((.*?)(\)|,).*$').search(
        caller.code_context[0]).groups()[0]
    if more:
      more_vars = re.compile(r'more=\[(.*?)\].*$').search(
          caller.code_context[0]).groups()[0]
      caller_more_vars = more_vars.split(',')

  the_class = ''
  if 'self' in stack.f_locals:
    the_class = stack.f_locals['self'].__class__.__name__
  header = '-DBG- {}:{}:{}:{} {} '.format(
      os.path.basename(caller.filename), the_class, caller.function,
      caller.lineno, message)

  info = '{}{}={}'.format(header, caller_var, tensor)
  for name, val in zip(caller_more_vars, more):
    info += ' {}={}'.format(name.strip(), val)
  tf.logging.info(info)

  if isinstance(tensor, tf.Tensor):
    tensors = []
    tensors += [tf.constant('{}.shape='.format(caller_var)), tf.shape(tensor)]
    for name, val in zip(caller_more_vars, more):
      tensors += [tf.constant('{}.shape='.format(name.strip())), tf.shape(val)]

    tensors += [tf.constant('{}='.format(caller_var)), tensor]
    for name, val in zip(caller_more_vars, more):
      tensors += [tf.constant('{}='.format(name.strip())), val]

    name = tensor.name if not tf.executing_eagerly() else '[eager]'
    info = '{}{} {}'.format(header, caller_var, name)
    return tf.identity(
        tf.Print(tensor, tensors, info, summarize=summarize),
        re.sub(':.*$', '', name))

  return tensor


def _Save(steps, prefix, key, val):
  filename = '%s.%08d.%s.npy' % (six.ensure_text(prefix), steps,
                                 six.ensure_text(key))
  with tf.io.gfile.GFile(filename, 'w') as outfile:
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
  if py_utils_flags.enable_asserts():
    return with_dependencies([tf.assert_equal(tf.rank(tensor), expected_rank)],
                             tensor)
  else:
    return tensor


def HasAtLeastRank(tensor, expected_rank):
  """Syntactic sugar for asserting that tensor has rank >= expected_rank."""
  if tensor.shape.ndims is not None and isinstance(expected_rank, int):
    assert tensor.shape.ndims >= expected_rank, (
        'Rank of tensor %d did not exceed the expected value %d.') % (
            tensor.shape.ndims, expected_rank)
    return tensor
  if py_utils_flags.enable_asserts():
    return with_dependencies(
        [tf.debugging.assert_greater_equal(tf.rank(tensor), expected_rank)],
        tensor)
  else:
    return tensor


def GetRank(tensor):
  """Returns tensor's rank as an int if it's available, otherwise a Tensor.

  Args:
    tensor: The input tensor.

  Returns:
    Either an int or a Tensor for the rank of the input tensor.
  """
  if tensor.shape.ndims is not None:
    return tensor.shape.ndims  # int
  else:
    return tf.rank(tensor)  # Tensor


def GetShape(tensor, ndims=None):
  """Returns tensor's shape as a list which can be unpacked, unlike tf.shape.

  Tries to return static shape if it's available. Note that this means
  some of the outputs will be ints while the rest will be Tensors.

  Args:
    tensor: The input tensor.
    ndims: If not None, returns the shapes for the first `ndims` dimensions.
  """
  tensor = tf.convert_to_tensor(tensor)
  dynamic_shape = tf.shape(tensor)

  # Early exit for unranked tensor.
  if tensor.shape.ndims is None:
    if ndims is None:
      return dynamic_shape
    else:
      return [dynamic_shape[x] for x in range(ndims)]

  # Ranked tensor.
  if ndims is None:
    ndims = tensor.shape.ndims
  else:
    ndims = min(ndims, tensor.shape.ndims)

  # Return mixture of static and dynamic dims.
  static_shape = tensor.shape.as_list()
  shapes = [
      static_shape[x] if static_shape[x] is not None else dynamic_shape[x]
      for x in range(ndims)
  ]
  return shapes


def HasShape(tensor, expected_shape, ndims=None):
  """Syntactic sugar for asserting that tensor has the expected shape.

  Args:
    tensor: A Tensor.
    expected_shape: A Python list or a 1D tensor. Elements of expected_shape can
      be -1 which indicate that any size is valid for that dimension.
    ndims: If not None, check only the first `ndims` dimensions of `tensor`.
      Must be equal to the length of `expected_shape` if not None.

  Returns:
    The input `tensor` with control dependencies that will raise a runtime
    error if dynamic shape checks fail.

  Raises:
    ValueError: A value error if the assertion fails at static shape checks.
  """
  if not py_utils_flags.enable_asserts():
    return tensor

  filepath, line, func, _ = traceback.extract_stack(limit=3)[-2]
  msg = 'LINGVO ASSERT %s:%s(%s)' % (re.sub(r'.*/', '',
                                               filepath), line, func)

  tensor_shape = GetShape(tensor)
  if ndims is not None:
    tensor_shape = tensor_shape[:ndims]

  # TODO(jngiam): Attempt to switch back to tf.Assert after it has better
  # support on GPUs.
  assert_op = ops.assert_shape_match(tensor_shape, expected_shape, msg=msg)

  # If expected_shape is a Tensor, then we are unable to perform static checks.
  # In this case, we can do a dynamic check and return.
  if isinstance(expected_shape, tf.Tensor):
    return with_dependencies([assert_op], tensor)

  # Infer ranks from the inputs.
  expected_rank = len(expected_shape)
  if isinstance(tensor_shape, tf.Tensor):
    tensor_rank = tensor.shape.ndims
  else:
    tensor_rank = len(tensor_shape)

  # If ndims is None, then either one of the ranks should not be None, or they
  # should both match. If both ranks are None, then they are both tensors and
  # should be caught by the earlier short-circuit.
  if ndims is None:
    if (tensor_rank is not None) and (expected_rank != tensor_rank):
      raise ValueError('Tensor does not match rank of expected shape.\n'
                       'Tensor shape: {} Expected shape: {}'.format(
                           tensor_shape, expected_shape))
    # Both tensors can be assumed to be of same rank.
    ndims = expected_rank
  else:
    if (tensor_rank is not None) and (tensor_rank < ndims):
      raise ValueError('Tensor has fewer dimensions than ndims.\n'
                       'Tensor shape: {} ndims: {}'.format(tensor_shape, ndims))
    if expected_rank != ndims:
      raise ValueError(
          'Expected shape must have number of dimensions equal to ndims.\n'
          'Expected shape: {} ndims: {}'.format(expected_shape, ndims))

  # Ensure that both tensor_shape and expected_shape are both lists.
  tensor_shape = tensor_shape[:ndims]
  if isinstance(tensor_shape, tf.Tensor):
    tensor_shape = tf.unstack(tensor_shape, num=ndims)

  # Map tf.Dimension values to their held values.
  tensor_shape = [
      v.value if isinstance(v, tf.Dimension) else v for v in tensor_shape
  ]
  expected_shape = [
      v.value if isinstance(v, tf.Dimension) else v for v in expected_shape
  ]

  all_static_checks = True
  for idx, (dim, expected_dim) in enumerate(zip(tensor_shape, expected_shape)):
    if isinstance(expected_dim, tf.Tensor):
      all_static_checks = False
    elif expected_dim == -1:
      continue
    elif isinstance(dim, tf.Tensor):
      all_static_checks = False
    elif dim != expected_dim:
      raise ValueError('Tensor does not match expected shape on dimension {}.\n'
                       'Tensor shape: {} Expected shape: {}'.format(
                           idx, tensor_shape, expected_shape))

  if all_static_checks:
    return tf.convert_to_tensor(tensor)
  else:
    return with_dependencies([assert_op], tensor)


def HasSameShape(x, ref):
  return HasShape(x, GetShape(ref))


def GetSize(tensor):
  shape = GetShape(tensor)
  if (isinstance(shape, tf.Tensor) or
      any([isinstance(x, tf.Tensor) for x in shape])):
    return tf.size(tensor)
  return np.prod(shape)


def CausalSelfAttenPadding(seqlen, dtype):
  """Wraps tf.linalg.band_part() for tflite compatibility."""
  if FLAGS.tflite_compatible:
    # [N, 1]
    rows = tf.expand_dims(tf.range(seqlen), -1)
    # [1, N]
    cols = tf.expand_dims(tf.range(seqlen), 0)
    row_cols = rows - cols
    return tf.where(row_cols < 0, tf.ones([seqlen, seqlen], dtype),
                    tf.zeros([seqlen, seqlen], tf.float32))
  else:
    return 1.0 - tf.linalg.band_part(
        tf.ones([seqlen, seqlen], dtype=dtype), -1, 0)


def outside_all_rewrites():  # pylint: disable=invalid-name
  return tf.control_dependencies(None)


# TODO(jamesqin): remove once b/147439702 is fixed.
_OUTSIDE_COMPILATION = threading.local()


def RunOnTpuHost(func, *args, **kwargs):
  r"""Runs the given function call on TPU host.

  Invokes func(\*args, \*\*kwargs) directly if not running on tpu.

  Args:
    func: the function to invoke.
    *args: args of func
    **kwargs: kwargs of func

  Returns:
    The function return value.
  """
  if use_tpu() and not getattr(_OUTSIDE_COMPILATION, 'on', False):
    _OUTSIDE_COMPILATION.on = True
    res = tf.tpu.outside_compilation(func, *args, **kwargs)
    _OUTSIDE_COMPILATION.on = False
  else:
    res = func(*args, **kwargs)
  return res


def tpu_host(func):  # pylint: disable=invalid-name
  r"""Decorates a python function to only run on TPU hosts.

  This function has no effect when running on CPU/GPU.

  Example::

    @py_utils.tpu_host()
    def ComputeWER(self):
      # Call a custom op computing WER.

  Args:
    func: the function to invoke

  Returns:
    A TPU-host only function
  """

  def Wrapped(*args, **kwargs):
    return RunOnTpuHost(func, *args, **kwargs)

  return Wrapped


# Maps a TPU job name ('/job:xxx') to the job's DeviceAssignment object.
# When there is only a single TPU job, the key could be None.
_tpu_device_assignment_dict = dict()


def SetTpuDeviceAssignment(tpu_device_assignment, job=None):
  if job in _tpu_device_assignment_dict:
    tf.logging.warning('tpu_device_assignment was already set, '
                       'overwriting with new assignment.')
  _tpu_device_assignment_dict[job] = tpu_device_assignment


# This function should called in unittest only.
def ClearTpuDevice():
  global _tpu_device_assignment_dict
  _tpu_device_assignment_dict = dict()


def GetTpuDeviceAssignment(job=None):
  return _tpu_device_assignment_dict[job]


# Whether it's running in eager mode.
_IS_EAGER_MODE = False


def SetIsEagerMode():
  global _IS_EAGER_MODE
  _IS_EAGER_MODE = True


def IsEagerMode():
  return _IS_EAGER_MODE


def SessionConfig(soft_placement=True,
                  inline=True,
                  cluster_def=None,
                  disable_meta_optimizer=False):
  """Returns a session config proto.

  Args:
    soft_placement: Turns allow_soft_placement on iff True.
    inline: Turns do_function_inlining on iff True.
    cluster_def: A tf.train.ClusterDef describing the cluster.
    disable_meta_optimizer: Turns off grappler/metagraph optimizer.

  Returns:
    A TF session config proto.
  """
  session_config = tf.config_pb2.ConfigProto(
      allow_soft_placement=soft_placement,
      graph_options=tf.GraphOptions(
          optimizer_options=tf.OptimizerOptions(
              opt_level=tf.OptimizerOptions.L1, do_function_inlining=inline)),
      cluster_def=cluster_def)
  session_config.share_cluster_devices_in_session = True

  if disable_meta_optimizer:
    # Useful if start-up time is critical.
    session_config.graph_options.rewrite_options.disable_meta_optimizer = True
  # Disable layout optimizer which increases GPU memory usage.
  session_config.graph_options.rewrite_options.layout_optimizer = (
      rewriter_config_pb2.RewriterConfig.OFF)
  return session_config


def AssertIsCompatible(a, b):
  assert a.IsCompatible(b), ('%s vs %s' % (a, b))


def SetShapes(dst_nmap, src_nmap):
  """Set shapes in dst_nmap using those in src_nmap."""
  AssertIsCompatible(src_nmap, dst_nmap)
  for src, dst in zip(src_nmap.Flatten(), dst_nmap.Flatten()):
    dst.set_shape(src.shape)


def Dtypes(nmap_list):
  """Returns all tensors' data types in a list."""
  return [v.dtype for v in Flatten(nmap_list)]


def Flatten(x):
  """Flattens 'x' by extracting tensors from nested structures to a list."""
  return tf.nest.flatten(x)


def Pack(tmpl, values):
  """Packs 'values' according to 'tmpl'."""
  return tf.nest.pack_sequence_as(tmpl, values)


def Transform(fn, *v):
  """Replaces every nested value x in 'v' with fn(x) and returns the result."""
  return tf.nest.map_structure(fn, *v)


def ConvertNoneGradientToZeros(xs, dxs):
  """Sanitize dxs so that None becomes zeros appropriately.

  Args:
    xs: A list of tensors.
    dxs: A list of tensors. dxs[i] corresponds to xs[i]'s gradient.

  Returns:
    A `.NestedMap` same as dxs with None replaced by a zero tensor.
  """
  fn = lambda x, dx: tf.zeros_like(x) if dx is None else dx
  return Transform(fn, xs, dxs)


def IsCompatible(lhs, rhs):
  """Returns true if lhs and rhs are compatible."""
  try:
    tf.nest.assert_same_structure(lhs, rhs)
    return True
  except (ValueError, TypeError):
    return False


class _Unique:
  """A helper to uniqify variables in a NestedMap."""

  def __init__(self):
    self._vset = set()

  def __call__(self, v):
    if (v is None) or (id(v) in self._vset):
      return False
    else:
      self._vset.add(id(v))
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

  class Wrapper:
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


def ToStaticShape(shape):
  """Converts 'shape' to a static shape."""
  if isinstance(shape, (list, tuple)):
    shape = [
        dim.value if isinstance(dim, tf.Dimension) else dim for dim in shape
    ]
    static_shape = []
    for dim in shape:
      if symbolic.IsExpr(dim):
        static_shape.append(symbolic.ToStatic(dim))
      else:
        static_shape.append(dim)
    return static_shape
  else:
    return shape.value if isinstance(shape, tf.Dimension) else shape


def Zeros(shape, *args, **kwargs):
  return tf.zeros(ToStaticShape(shape), *args, **kwargs)


class UniformSampler:
  """A reservoir sampler.

  This class implements reservoir sampling: Given a limit of `num_samples` total
  samples, this class maintains a uniform probability (1 / `num_samples`) of
  keeping any item dynamically added to the sampler.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for details.
  """

  def __init__(self, num_samples):
    assert num_samples > 0
    self._num_samples = num_samples
    self._num_seen_items = 0
    self._samples = []

  def Add(self, item):
    """Add item to sampler."""
    self._num_seen_items += 1

    if len(self._samples) < self._num_samples:
      self._samples.append(item)
      return

    index = np.random.randint(0, self._num_seen_items)
    if index < self._num_samples:
      self._samples[index] = item

  @property
  def samples(self):
    """Fetch the current samples from the sampler."""
    return self._samples


class RNNCellStateInit:
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


def InitRNNCellState(shape, init=None, dtype=None, name=None, is_eval=False):
  """Initial state definitions for RNN cell implementations.

  Args:
    shape: A array of ints/symbols for specifying the shape of the state.
    init: Hyperparameters as returned by one of the static implemetaitons in
      RNNCellStateInit.
    dtype: The dype of the states. Defaults to tf.float32.
    name: A name for the operation. If --stateless_vars_init is set, this name
      is used to generate a seed on a per-variable basis. Otherwise, this name
      is optional.
    is_eval: Bool, set to True if we need special behavior in eval mode.

  Returns:
    A Tensor of the specified shape, and sampled from the distribution as
    defined by the init parameters.
  """
  shape = ToStaticShape(shape)

  if init is None:
    init = DefaultRNNCellStateInit()
  if dtype is None:
    dtype = tf.float32

  method = init.method
  if ((method in ['zeros']) or (method in ['random_normal'] and is_eval)):
    init_state = tf.zeros(shape=shape, dtype=dtype, name=name)
  elif method in ['random_normal']:
    if use_stateless_vars_init():
      if name is None:
        raise ValueError('InitRNNCellState() requires a `name` argument when '
                         '--stateless_vars_init is enabled.')
      seed = _GenerateStatelessRngSeed(name, init.seed)
      init_state = stateless_random_ops.stateless_random_normal(
          shape=shape, dtype=dtype, name=name, seed=seed)
    else:
      init_state = tf.random.normal(
          shape=shape, dtype=dtype, name=name, seed=init.seed)
  else:
    raise ValueError('Initialization method (%s) not supported.' % method)

  return init_state


class WeightInit:
  """Static class providing weight initialization config params."""

  @staticmethod
  def _Params(method, scale, seed):
    """Parameters of this class."""
    p = hyperparams.Params()
    p.Define('method', method, 'Initialization method.')
    p.Define('scale', scale, 'Initialization scale.')
    p.Define('seed', seed, 'Random seed used to generate initial values.')
    p.Freeze()
    return p

  @staticmethod
  def Gaussian(scale=1.0, seed=None):
    """scale * tf.random.normal(0, 1.0)."""
    return WeightInit._Params('gaussian', scale, seed)

  @staticmethod
  def Uniform(scale=1.0, seed=None):
    """scale * tf.random.uniform(-1.0, 1.0)."""
    return WeightInit._Params('uniform', scale, seed)

  @staticmethod
  def UniformPositive(scale=1.0, seed=None):
    """scale * tf.random.uniform(0., 1.0)."""
    return WeightInit._Params('uniform_positive', scale, seed)

  @staticmethod
  def Category(scale=2, seed=None):
    """tf.floor(scale * tf.random.uniform(0., 1.0))."""
    return WeightInit._Params('category', scale, seed)

  @staticmethod
  def Xavier(scale=1.0, seed=None):
    """Xavier initialization (x = sqrt(6. / (in + out)); [-x, x])."""
    return WeightInit._Params('xavier', scale, seed)

  @staticmethod
  def XavierWithFixupParams(scale=1.0,
                            depth=1.0,
                            layers_per_residual_block=1.0,
                            seed=None):
    """Xavier initialization with Fixup."""
    scale = scale * math.pow(depth, (-1.0 / (2 * layers_per_residual_block)))
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
    """scale * tf.random.truncated_normal(0, 1.0)."""
    return WeightInit._Params('truncated_gaussian', scale, seed)

  @staticmethod
  def GaussianSqrtDim(scale=1.0, seed=None):
    """scale * tf.random.normal(0, 1 / sqrt(dim0))."""
    return WeightInit._Params('gaussian_sqrt_dim', scale, seed)

  @staticmethod
  def GaussianSqrtFanIn(scale=1.0, seed=None):
    """scale * tf.random.normal(0, 1 / sqrt(fan_in))."""
    return WeightInit._Params('gaussian_sqrt_fanin', scale, seed)

  @staticmethod
  def GaussianSqrtFanOut(scale=1.0, seed=None):
    """scale * tf.random.normal(0, 1 / sqrt(fan_out))."""
    return WeightInit._Params('gaussian_sqrt_fanout', scale, seed)

  @staticmethod
  def GaussianSqrtFanAvg(scale=1.0, seed=None):
    """tf.random.normal(0, sqrt(2.0 / (in + out)))."""
    return WeightInit._Params('gaussian_sqrt_fanavg', scale, seed)

  @staticmethod
  def UniformSqrtDim(scale=1.0, seed=None):
    """scale * tf.uniform(-1 / sqrt(dim0), 1 / sqrt(dim0))."""
    return WeightInit._Params('uniform_sqrt_dim', scale, seed)

  @staticmethod
  def UniformUnitScaling(scale=1.0, seed=None):
    """scale * sqrt(3) / sqrt(dim0) * tf.uniform(-1, 1)."""
    return WeightInit._Params('uniform_unit_scaling', scale, seed)

  @staticmethod
  def UniformUnitScalingFanAvg(scale=1.0, seed=None):
    """Same as tf.variance_scaling_initializer() ...

    Samples are drawn from a uniform distribution within [-limit, limit], with
    limit = sqrt(3 * scale / n)

    where
    n = max(1., (fan_in + fan_out) / 2).
    See tf.keras.initializers.VarianceScaling for details.

    Args:
      scale: A Python float.
      seed: A Python int or None.

    Returns:
      A WeightInit param.
    """
    return WeightInit._Params('uniform_unit_scaling_fan_avg', scale, seed)

  @staticmethod
  def TruncatedGaussianSqrtDim(scale=1.0, seed=None):
    """scale * tf.random.truncated_normal(0, 1 / sqrt(dim0))."""
    return WeightInit._Params('truncated_gaussian_sqrt_dim', scale, seed)

  @staticmethod
  def TruncatedGaussianSqrtFanIn(scale=1.0, seed=None):
    """scale * tf.random.truncated_normal(0, 1 / sqrt(fan_in))."""
    return WeightInit._Params('truncated_gaussian_sqrt_fanin', scale, seed)

  @staticmethod
  def TruncatedGaussianSqrtFanOut(scale=1.0, seed=None):
    """scale * tf.random.truncated_normal(0, 1 / sqrt(fan_out))."""
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


# TODO(rpang, jonathanasdf): explore adding _is_default to hyperparams.Param.
def IsDefaultParamInit(p):
  return (p.method == 'xavier' and
          abs(p.scale - _DEFAULT_XAVIER_INIT) < 1e-7 and p.seed is None)


def WeightParams(shape,
                 init=None,
                 dtype=None,
                 collections=None,
                 device_mesh=None,
                 tensor_split_dims_mapping=None):
  """Returns a hyperparams for a weight variable given the shape/init/dtype."""
  if init is None:
    init = WeightInit.Xavier(_DEFAULT_XAVIER_INIT)
  if dtype is None:
    dtype = tf.float32
  if collections is None:
    collections = []
  if device_mesh is not None:
    assert tensor_split_dims_mapping is not None
    assert len(tensor_split_dims_mapping) == len(shape)

  p = hyperparams.Params()
  p.Define('dtype', dtype, 'The weight data type.')
  p.Define('shape', shape, 'The weight shape.')
  p.Define('init', init, 'Initialization method.')
  p.Define('collections', collections,
           'Variable collections this weight belongs to.')
  p.Define(
      'device_mesh', device_mesh,
      'A numpy.ndarray describing the topology of a device mesh to partition'
      ' this variable onto. Each element in the np.ndarray is the ID of a'
      ' device in the topology. device_mesh and tensor_split_dims_mapping below'
      ' together specifies how this weight tensor should be sharded across'
      ' different tpu cores. If None, this variable is not sharded.'
      ' Here are examples: np.array([0, 1, 2, 3, 4, 5, 6, 7]) which is a 1d'
      ' mesh with 8 devices, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]) which is'
      ' 2d matrix of 8 devices.')
  p.Define(
      'tensor_split_dims_mapping', tensor_split_dims_mapping,
      'A list of integers that map each tensor axis to the device mesh axis'
      ' along which it is sharded. Its length is the tensor rank, and'
      ' split_dims_mapping[i] is device mesh axis for tensor dimension i. Use'
      ' -1 for tensor dimensions that are not sharded. If the list is set to'
      ' None and a device_mesh is specified, the sharding will be treated as'
      ' replicated. Here is a concrete examples: '
      '   device_mesh=np.array([[0, 1, 2, 3] [4, 5, 6, 7]]), of shape [2, 4]'
      '   shape=[x, y, z], so this is a 3d variable.'
      '   tensor_split_dims_mapping=[-1, -1, 1], in this case, the third dim'
      '   of the variable is split along the second dim of the mesh. Each '
      '   split of the variable is of the shape [x, y, z/4].')
  return p


def FindNeeded(endpoints):
  """List names of tensors and operations required to compute endpoints."""
  names_seen = set()
  queue = []
  for e in Flatten(endpoints):
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


class _CollectionGetter:
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


def SanitizeScopeKey(key):
  """Removes invalid symbols from name_scope keys."""
  if key.startswith('_'):
    key = key[1:]
  return key.replace('[', '_').replace(']', '')


# Maintain a session for unit tests (initialized in test_utils.py).
_SESSION_SCOPE = ThreadLocalStack()


@contextlib.contextmanager
def UnitTestSessionScope(sess):
  _SESSION_SCOPE.stack.append(sess)
  try:
    yield
  finally:
    _SESSION_SCOPE.stack.pop()


def GetUnitTestSession():
  """Get the current variable reuse setting."""
  return _SESSION_SCOPE.stack[-1] if _SESSION_SCOPE.stack else None


# Global variable to control multitask variable reuse
# If False (default) the default tf.get_variable is used, that is:
# - Reusing scopes only allow getting existing variables
# - Non-reusing scopes only allow getting new variables
# With GetOpportunisticVariableReuse() == True:
# - Reusing scopes only allow getting existing variables, as usual
# - Non-reusing scopes reuse new variables or get new ones
_OPPORTUNISTIC_VARIABLE_REUSE = ThreadLocalStack()


@contextlib.contextmanager
def OpportunisticVariableReuseScope(enable_opportunistic_reuse=True):
  _OPPORTUNISTIC_VARIABLE_REUSE.stack.append(enable_opportunistic_reuse)
  try:
    yield
  finally:
    _OPPORTUNISTIC_VARIABLE_REUSE.stack.pop()


def GetOpportunisticVariableReuse():
  """Get the current variable reuse setting."""
  return (_OPPORTUNISTIC_VARIABLE_REUSE.stack[-1]
          if _OPPORTUNISTIC_VARIABLE_REUSE.stack else False)


_VARIABLE_RENAME_RULES = ThreadLocalStack()

# Global variable to track task calling scope.
# Currently only used for TPU Embedding purposes as a TPUEmbeddinglayer
# may be shared across tasks and the calling task needs to be known
# for tracking embedding activations for backprop.
_TASK_CALL_SCOPE = ThreadLocalStack()


@contextlib.contextmanager
def TaskCallScope(task_name):
  _TASK_CALL_SCOPE.stack.append(task_name)
  try:
    yield
  finally:
    _TASK_CALL_SCOPE.stack.pop()


def GetTaskCallScope():
  """Get the current task call scope."""
  return _TASK_CALL_SCOPE.stack[-1] if _TASK_CALL_SCOPE.stack else None


@contextlib.contextmanager
def VariableRenameScope(renames):
  """Append the renaming rules to the stack of renames.

  Args:
    renames: pairs of (regexp, new_name_format). If the regexp matches, the
      new_name_format will be interpolated using the matched groups.

  Yields:
    scope in which the renaming rules are applied
  """
  _VARIABLE_RENAME_RULES.stack.append(renames)
  try:
    yield
  finally:
    _VARIABLE_RENAME_RULES.stack.pop()


def GetVariableName(name):
  """Get variable name after application of all renaming rules.

  Args:
    name: untransformed variable name with scope_name prepended

  Returns:
    name possibly modified using renaming rules
  """
  matched = False
  new_name = name
  for renames in _VARIABLE_RENAME_RULES.stack:
    for regexp, name_format in renames:
      match = re.match(regexp, name)
      if match:
        if matched:
          tf.logging.warning('Multiple matches for: %s', name)
        matched = True
        new_name = name_format % match.groups()
  if new_name != name:
    tf.logging.info("WARNING!!! Renaming variable '%s' to '%s'", name, new_name)
  return new_name


def GenerateSeedFromName(name):
  """Generate a random seed from a name string.

  Args:
    name: A string.

  Returns:
    An integer seed in the range [0, 2**31 - 1).
  """
  md5 = hashlib.md5()
  md5.update(six.ensure_binary(name))
  return np.int64(int(md5.hexdigest(), 16) % (2**31 - 1))


def MaybeGenerateSeedFromScope():
  """Generate a random seed from the current name of the scope.

  If running in eager mode, this returns 0.

  Returns:
    An integer seed in the range [0, 2**31 - 1).
  """
  if not tf.executing_eagerly():
    return GenerateSeedFromName(tf.no_op(name='new_step_seed').name)
  return 0


def GenerateSeedFromId(obj_id):
  """Generate a random seed from the id of an object.

  If deterministic execution (i.e. unit test), generate the seed from a fixed
  unique name instead.

  Args:
    obj_id: id(object).

  Returns:
    An integer seed in the range [0, 2**31 - 1).
  """
  if tf.get_default_graph().seed is not None:
    # We are in a program/test which need determistic randomization.
    with tf.name_scope(''):
      return GenerateSeedFromName(tf.no_op(name='new_step_seed').name)

  md5 = hashlib.md5()
  md5.update(np.int64(obj_id))
  return np.int64(int(md5.hexdigest(), 16) % (2**31 - 1))


# To keep track of all the variables ever gets created by the CreateVariable
# routine below.
_ALL_VARS_KEY = ('__lingvo_all_vars',)

_get_all_vars = _CollectionGetter(_ALL_VARS_KEY, lambda: {})

_VARIABLE_SHAPE_PREFIXES = ThreadLocalStack()


@contextlib.contextmanager
def VariableShapePrefixContext(shape_prefix):
  """Add a shape prefix to variable created by CreateVariable().

  Args:
    shape_prefix: a positive integer of shape prefix.

  Yields:
    None.
  """
  assert shape_prefix > 0, ('%s' % shape_prefix)
  _VARIABLE_SHAPE_PREFIXES.stack.append(shape_prefix)
  try:
    yield
  finally:
    _VARIABLE_SHAPE_PREFIXES.stack.pop()


def GetVariableShapePrefixes():
  """Return the list of shape prefixes for CreateVariable()."""
  return _VARIABLE_SHAPE_PREFIXES.stack


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


_VARIABLE_CREATOR_STACK = ThreadLocalStack().stack


def _DefaultVariableCreator(**kwargs):
  kwargs.pop('var_name', None)
  kwargs.pop('var_params', None)
  return tf.get_variable(**kwargs)


def _GetVariableCreator():
  fn = _DefaultVariableCreator
  for wrapper in reversed(_VARIABLE_CREATOR_STACK):
    fn = functools.partial(wrapper, fn)
  return fn


@contextlib.contextmanager
def VariableCreatorScope(variable_creator):
  """Yields a context around a variable_creator, used by `CreateVariable()`.

  The function must have the following signature::

    def variable_creator(next_creator, **kwargs)

  The function may delegate variable creation to the next variable creator, or
  return its own tf.Variable.

  This differs from tf.variable_creator_scope in that tf.variable_creator_scope
  modifies a tf.Variable() call while this modifies a tf.get_variable() call. As
  the code is migrated to TF2 and tf.get_variable() is deprecated, this may be
  upgraded to using tf.variable_creator_scope instead.

  This differs from tf.variable_scope(custom_getter=variable_creator) in that
  the kwargs passed can be manipulated.

  Variable creators are resolved from the outermost towards the innermost.

  The innermost variable creator function is tf.get_variable.

  The passed in kwargs must conform to what tf.get_variable accepts, with the
  addition of `var_name` and `var_params`.

  Args:
    variable_creator: A variable creator function.
  """
  _VARIABLE_CREATOR_STACK.append(variable_creator)
  try:
    yield
  finally:
    _VARIABLE_CREATOR_STACK.pop()


def PlaceOnTpuCore(core_id):
  """Returns a VariableCreatorScope that places variables on a given tpu core.

  Only applies when running with TPUs.

  Does not yet properly support model parallelism.

  Args:
    core_id: The tpu core id.
  """

  def Creator(next_creator, **kwargs):
    cluster = cluster_factory.Current()
    if use_tpu():
      device = cluster.WorkerDeviceInModelSplit(core_id)
    elif (
        tpu_compat() and
        cluster.params.job in ('controller', 'trainer_client', 'executor_tpu')):
      # The job is running in a fleet that uses tpu, but does not itself have
      # access to the tpu, e.g. controller job. In this case, the returned
      # device needs to be the cpu device on the tpu host for the given core.
      # FIXME: the current implementation is wrong for large values of core_id.
      device = cluster.ListDevices(cluster.params.worker)[0, 0]
    else:
      device = ''

    with tf.device(device):
      return next_creator(**kwargs)

  return VariableCreatorScope(Creator)


# TODO(yonghui): Add support for partitioned Variables.
def CreateVariable(name,
                   params,
                   reuse=None,
                   trainable=True,
                   collections=None,
                   default_seed=None,
                   synchronization=tf.VariableSynchronization.AUTO,
                   aggregation=tf.VariableAggregation.NONE):
  """Creates tf.Variable according to param_config.

  Args:
    name: A string, name of the variable.
    params: A WeightParams specifying the details of how this variable should be
      constructed and initialized.
    reuse: Whether or not to reuse an existing variable. It has the same
      semantics as the reuse arg in tf.variable_scope.
    trainable: Whether or not the variable is trainable.
    collections: Override the default variable collection (
      tf.GraphKeys.GLOBAL_VARIABLES).
    default_seed: Seed to use for initialization if not specified in params.
      Used for deterministic initialization in tests.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      tf.VariableSynchronization. By default the synchronization is set to AUTO
      and the current DistributionStrategy chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class tf.VariableAggregation.

  Returns:
    The created variable.
  """
  if use_stateless_vars_init():
    return _CreateVariableStateless(name, params, reuse, trainable, collections,
                                    default_seed, synchronization, aggregation)
  else:
    return _CreateVariableStateful(name, params, reuse, trainable, collections,
                                   default_seed, synchronization, aggregation)


def _CreateVariableStateful(name,
                            params,
                            reuse=None,
                            trainable=True,
                            collections=None,
                            default_seed=None,
                            synchronization=tf.VariableSynchronization.AUTO,
                            aggregation=tf.VariableAggregation.NONE):
  """Creates tf.Variable using TF stateful RNGs according to param_config.

  Args:
    name: A string, name of the variable.
    params: A WeightParams specifying the details of how this variable should be
      constructed and initialized.
    reuse: Whether or not to reuse an existing variable. It has the same
      semantics as the reuse arg in tf.variable_scope.
    trainable: Whether or not the variable is trainable.
    collections: Override the default variable collection (
      tf.GraphKeys.GLOBAL_VARIABLES).
    default_seed: Seed to use for initialization if not specified in params.
      Used for deterministic initialization in tests.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      tf.VariableSynchronization. By default the synchronization is set to AUTO
      and the current DistributionStrategy chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class tf.VariableAggregation.

  Returns:
    The created variable.
  """
  p = params.Copy()
  shape = tf.TensorShape(ToStaticShape(p.shape)).as_list()
  if shape:
    assert all([dim_size > 0 for dim_size in shape]), shape
    dim0 = shape[0]
  else:
    dim0 = 1
  assert p.init.method == 'constant' or np.all(np.asarray(p.init.scale) >= 0)
  method = p.init.method
  scale = p.init.scale
  seed = p.init.seed

  if IsDefaultParamInit(p.init):
    tf.logging.warning(
        'WARNING!!! var %s is using the default xavier initializer.'
        ' Make sure this is intended.', name)

  with tf.variable_scope(name) as scope:
    var_name = GetVariableName(scope.name)

  if tf.get_default_graph().seed is not None:
    # We are in a program/test which need determistic randomization.
    if seed is None:
      if default_seed is not None:
        seed = default_seed
      else:
        # We are not given a per-variable random seed. We use hash of
        # variable name as a stable random seed.
        seed = GenerateSeedFromName(var_name)

  init_dtype = p.dtype.real_dtype

  # TODO(b/172827074): we do not natively support var initialization for
  # int8 type except for constant initialization.
  # NOTE: For int8, we initialize by scaling float32 random values to integer.
  if init_dtype == tf.int8:
    init_dtype = tf.float32

  v_init = _CreateVarInitStateful(name, method, shape, dim0, seed, scale,
                                  init_dtype)

  if p.dtype == tf.complex64:

    def ComplexWrapper(init):

      def _Wrapper(shape, dtype, partition_info):
        del dtype
        # A more complex alternative may be to use the init function for
        # magnitudes and uniform random for phases instead.
        shape = [2] + shape
        value = init(shape, init_dtype, partition_info)
        return tf.complex(value[0], value[1])

      return _Wrapper

    v_init = ComplexWrapper(v_init)

  if p.dtype == tf.int8:

    def FloatToInt8Wrapper(init):

      def _Wrapper(shape, dtype, partition_info):
        del dtype
        value = init(shape, init_dtype, partition_info)
        scale = tf.math.maximum(
            tf.math.reduce_min(value) / -127,
            tf.math.reduce_max(value) / 127)
        value = tf.divide(value, scale)
        return tf.cast(value, tf.int8)

      return _Wrapper

    v_init = FloatToInt8Wrapper(v_init)

  # Variable creators.
  def MaybePinVarsToCpu(next_creator, **kwargs):
    if _FromGlobal('pin_vars_to_cpu'):
      with tf.device('/cpu:0'):
        return next_creator(**kwargs)
    return next_creator(**kwargs)

  def MaybeOpportunisticVariableReuse(next_creator, **kwargs):
    try:
      return next_creator(**kwargs)
    except ValueError:  # Possibly the variable already exists
      if GetOpportunisticVariableReuse():
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
          return next_creator(**kwargs)
      else:
        raise

  def LingvoVariableCreator(next_creator, **kwargs):
    """Lingvo variable creator."""
    # TODO(yonghui): Possibly get away from variable_scope and implement our own
    # variable sharing mechanism.
    with tf.variable_scope(name) as scope:
      var_scope = tf.VariableScope(
          scope.reuse,
          custom_getter=scope.custom_getter,
          caching_device=scope.caching_device,
          use_resource=True)
    with tf.variable_scope(var_scope), tf.variable_scope(var_name, reuse=reuse):
      var = next_creator(**kwargs)

    var_ref = var.experimental_ref()  # For key in dict/set.
    all_vars = _get_all_vars()
    if var_ref in all_vars:
      tf.logging.info('Reusing var %s', var.name)
      cached = all_vars[var_ref]
      assert cached == p.ToText(), ('Cached config:\n %s vs new config:\n %s' %
                                    (cached, p.ToText()))
    else:
      tf.logging.info('Creating var %s shape=%s on device %s', var.name,
                      var.shape, var.device)
      all_vars[var_ref] = p.ToText()
      for col in p.collections:
        tf.add_to_collection(col, var)
    return var

  with VariableCreatorScope(LingvoVariableCreator):
    with VariableCreatorScope(MaybeOpportunisticVariableReuse):
      with VariableCreatorScope(MaybePinVarsToCpu):
        var = _GetVariableCreator()(
            var_name=var_name,
            var_params=p,
            name='var',
            shape=GetVariableShapePrefixes() + list(shape),
            dtype=p.dtype,
            initializer=v_init,
            collections=collections,
            trainable=trainable,
            validate_shape=True,
            synchronization=synchronization,
            aggregation=aggregation)

  # Shard the variable according to the sharding spec.
  tensor_split_dims_mapping = p.tensor_split_dims_mapping
  if tensor_split_dims_mapping is not None:
    tensor_split_dims_mapping = ([-1] * len(GetVariableShapePrefixes()) +
                                 tensor_split_dims_mapping)
  var = gshard_utils.MeshSplit(
      var, p.device_mesh, tensor_split_dims_mapping, use_sharding_op=False)

  return var


def _CreateVariableStateless(name,
                             params,
                             reuse=None,
                             trainable=True,
                             collections=None,
                             default_seed=None,
                             synchronization=tf.VariableSynchronization.AUTO,
                             aggregation=tf.VariableAggregation.NONE):
  """Creates tf.Variable using TF stateless RNGs according to `params`.

  Args:
    name: A string, name of the variable.
    params: A WeightParams specifying the details of how this variable should be
      constructed and initialized.
    reuse: Whether or not to reuse an existing variable. It has the same
      semantics as the reuse arg in tf.variable_scope.
    trainable: Whether or not the variable is trainable.
    collections: Override the default variable collection (
      tf.GraphKeys.GLOBAL_VARIABLES).
    default_seed: Seed to use for initialization if not specified in params.
      Used for deterministic initialization in tests.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      tf.VariableSynchronization. By default the synchronization is set to AUTO
      and the current DistributionStrategy chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class tf.VariableAggregation.

  Returns:
    The created variable.
  """
  p = params.Copy()
  shape = tf.TensorShape(ToStaticShape(p.shape)).as_list()
  if shape:
    assert all([dim_size > 0 for dim_size in shape]), shape
    dim0 = shape[0]
  else:
    dim0 = 1
  assert p.init.method == 'constant' or np.all(np.asarray(p.init.scale) >= 0)
  method = p.init.method
  scale = p.init.scale
  seed = p.init.seed

  if IsDefaultParamInit(p.init):
    tf.logging.warning(
        'WARNING!!! var %s is using the default xavier initializer.'
        ' Make sure this is intended.', name)

  with tf.variable_scope(name) as scope:
    var_name = GetVariableName(scope.name)

  user_seed = seed if seed is not None else default_seed
  seed = _GenerateStatelessRngSeed(var_name, user_seed)

  init_dtype = p.dtype.real_dtype
  v_init = _CreateVarInitStateless(name, method, shape, dim0, seed, scale,
                                   init_dtype)

  if p.dtype == tf.complex64:
    raise TypeError(
        'Stateless variable initialization does not support tf.complex64.')

  def LingvoVariableCreator(next_creator, **kwargs):
    """Lingvo variable creator."""
    # TODO(yonghui): Possibly get away from variable_scope and implement our own
    # variable sharing mechanism.
    with tf.variable_scope(name) as scope:
      var_scope = tf.VariableScope(
          scope.reuse,
          custom_getter=scope.custom_getter,
          caching_device=scope.caching_device,
          use_resource=True)
    with tf.variable_scope(var_scope), tf.variable_scope(var_name, reuse=reuse):
      var = next_creator(**kwargs)

    var_ref = var.experimental_ref()  # For key in dict/set.
    all_vars = _get_all_vars()
    if var_ref in all_vars:
      tf.logging.info('Reusing var %s', var.name)
      cached = all_vars[var_ref]
      assert cached == p.ToText(), ('Cached config:\n %s vs new config:\n %s' %
                                    (cached, p.ToText()))
    else:
      tf.logging.info('Creating var %s shape=%s on device %s', var.name,
                      var.shape, var.device)
      all_vars[var_ref] = p.ToText()
      for col in p.collections:
        tf.add_to_collection(col, var)
    return var

  with VariableCreatorScope(LingvoVariableCreator):
    var = _GetVariableCreator()(
        var_name=var_name,
        var_params=p,
        name='var',
        shape=GetVariableShapePrefixes() + list(shape),
        dtype=p.dtype,
        initializer=v_init,
        collections=collections,
        trainable=trainable,
        validate_shape=True,
        synchronization=synchronization,
        aggregation=aggregation)

  # Shard the variable according to the sharding spec.
  tensor_split_dims_mapping = p.tensor_split_dims_mapping
  if tensor_split_dims_mapping is not None:
    tensor_split_dims_mapping = ([-1] * len(GetVariableShapePrefixes()) +
                                 tensor_split_dims_mapping)
  var = gshard_utils.MeshSplit(
      var, p.device_mesh, tensor_split_dims_mapping, use_sharding_op=False)

  return var


def _RandomXavierUniformInitializer(method, scale, seed):
  """Creates a random Xavier uniform initializer."""

  def XavierUniform(shape, dtype, partition_info):
    """Xavier initialization (x = sqrt(6. / (in + out)); scale*[-x, x])."""
    del partition_info  # Unused.
    if not shape:
      raise ValueError('\'shape\' must not be \'None\' or 0 for XavierUniform')
    fan_in, fan_out = GetFanInFanOut(shape)
    if method == 'xavier':
      limit = math.sqrt(6. / (fan_in + fan_out))
    elif method == 'geo_mean_xavier':
      limit = math.sqrt(3. / math.sqrt(fan_in * fan_out))
    return scale * tf.random.uniform(shape, -limit, limit, dtype, seed)

  return XavierUniform


def _CreateVarInitStateful(name, method, shape, dim0, seed, scale, init_dtype):
  """Creates variable initialization function for a stateful RNG."""
  if (method in [
      'gaussian_sqrt_dim', 'uniform_sqrt_dim', 'truncated_gaussian_sqrt_dim'
  ]):
    if len(shape) > 2:
      # This is probably not the right method to use when len(shape) > 2,
      # e.g. dim0 will be 3 with a 3x3 conv2d kernel.
      tf.logging.warning(
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
  if method in ['gaussian_sqrt_fanavg']:
    fan_in, fan_out = GetFanInFanOut(shape)
    if fan_in is not None and fan_out is not None:
      scale *= math.sqrt(2.0 / (fan_in + fan_out))

  if method in [
      'gaussian', 'gaussian_sqrt_dim', 'gaussian_sqrt_fanin',
      'gaussian_sqrt_fanout', 'gaussian_sqrt_fanavg'
  ]:
    v_init = init_ops.random_normal_initializer(
        mean=0.0, stddev=scale, seed=seed, dtype=init_dtype)
  elif method in ['uniform', 'uniform_sqrt_dim']:
    v_init = init_ops.random_uniform_initializer(
        minval=-scale, maxval=scale, seed=seed, dtype=init_dtype)
  elif method in ['uniform_positive']:
    v_init = init_ops.random_uniform_initializer(
        minval=0.0, maxval=scale, seed=seed, dtype=init_dtype)
  elif method == 'category':
    uniform_init = init_ops.random_uniform_initializer(
        minval=0.0, maxval=scale, seed=seed, dtype=init_dtype)
    v_init = lambda *args, **kwargs: tf.floor(uniform_init(*args, **kwargs))
  elif method in ['uniform_unit_scaling']:
    v_init = init_ops.uniform_unit_scaling_initializer(
        factor=scale, seed=seed, dtype=init_dtype)
  elif method in ['uniform_unit_scaling_fan_avg']:
    v_init = tf.variance_scaling_initializer(
        scale=scale,
        mode='fan_avg',
        distribution='uniform',
        seed=seed,
        dtype=init_dtype)
  elif method in [
      'truncated_gaussian', 'truncated_gaussian_sqrt_dim',
      'truncated_gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanout'
  ]:
    v_init = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=scale, seed=seed, dtype=init_dtype)
  elif method in ['constant']:
    v_init = init_ops.constant_initializer(value=scale, dtype=init_dtype)
  elif method in ['xavier', 'geo_mean_xavier']:

    def XavierUniform(shape, dtype, partition_info):
      """Xavier initialization (x = sqrt(6. / (in + out)); scale*[-x, x])."""
      del partition_info  # Unused.
      if not shape:
        raise ValueError(
            '\'shape\' must not be \'None\' or 0 for XavierUniform')
      fan_in, fan_out = GetFanInFanOut(shape)
      if method == 'xavier':
        limit = math.sqrt(6. / (fan_in + fan_out))
      elif method == 'geo_mean_xavier':
        limit = math.sqrt(3. / math.sqrt(fan_in * fan_out))
      return scale * tf.random.uniform(shape, -limit, limit, dtype, seed)

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
    v_init = init_ops.random_uniform_initializer(
        minval=-bound, maxval=bound, seed=seed, dtype=init_dtype)
  else:
    assert False, 'init_type `%s` not supported.' % method

  return v_init


def _GenerateStatelessRngSeed(name, seed):
  """Generates a 2-tuple seed for a stateless variable initializer.

  We want to ensure that different variables end up with different random values
  even when they are passed the same seed and shape. To this aim, this function
  generates a pseudo-unique seed by hashing the variable name and mapping it
  into a scalar seed. More specifically, the returned value is a 2-tuple of
  tf.int32 scalar, where the first element is the user-provided seed and the
  second element is obtained by hashing the variable name.

  Args:
    name: The variable name for which to generate a stateless-like seed.
    seed: The user-specified scalar seed.

  Returns:
    A 2-tuple seed of tf.int32 values (for TPU compatibility).
  """
  seed0 = seed or 0
  seed1 = GenerateSeedFromName(name)
  return tf.constant([seed0, seed1], dtype=tf.int32)


def _DeterministicRandomNormalInitializer(seed, mean, stddev):
  """Creates a random normal initializer."""

  def DeterministicNormal(shape, dtype, partition_info):
    del partition_info  # Unused.
    return stateless_random_ops.stateless_random_normal(
        shape=shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)

  return DeterministicNormal


def _DeterministicRandomUniformInitializer(seed, minval, maxval):
  """Creates a random uniform initializer."""

  def DeterministicUniform(shape, dtype, partition_info):
    del partition_info  # Unused.
    return stateless_random_ops.stateless_random_uniform(
        shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype)

  return DeterministicUniform


def _DeterministicRandomTruncatedNormalInitializer(seed, mean, stddev):
  """Creates a random truncated normal initializer."""

  def DeterministicTruncatedNormal(shape, dtype, partition_info):
    del partition_info  # Unused.
    return stateless_random_ops.stateless_truncated_normal(
        shape=shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)

  return DeterministicTruncatedNormal


def _DeterministicRandomUniformUnitScalingInitializer(seed, factor):
  """Creates a random uniform unit scaling initializer."""

  def DeterministicUniformUnitScaling(shape, dtype, partition_info):
    # The following logic is originally from (UniformUnitScaling.__call__())
    # in TensorFlow: python/ops/init_ops.py
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape

    input_size = 1.0
    # Estimating input size is not possible to do perfectly, but we try.
    # The estimate, obtained by multiplying all dimensions but the last one,
    # is the right thing for matrix multiply and convolutions (see above).
    for dim in scale_shape[:-1]:
      input_size *= float(dim)
    # Avoid errors when initializing zero-size tensors.
    input_size = max(input_size, 1.0)
    maxval = math.sqrt(3 / input_size) * factor
    return stateless_random_ops.stateless_random_uniform(
        shape=shape, seed=seed, minval=-maxval, maxval=maxval, dtype=dtype)

  return DeterministicUniformUnitScaling


def _DeterministicRandomVarianceScalingInitializer(scale, mode, distribution,
                                                   seed):
  """Creates a variance scaling initializer."""

  if scale <= 0.:
    raise ValueError('`scale` must be positive float.')
  if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
    raise ValueError('Invalid `mode` argument:', mode)
  distribution = distribution.lower()
  if distribution not in {
      'normal', 'uniform', 'truncated_normal', 'untruncated_normal'
  }:
    raise ValueError('Invalid `distribution` argument:', distribution)

  def DeterministicVarianceScaling(shape, dtype, partition_info):
    # This is originally from TensorFlow: python/ops/init_ops.py
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    # Handle special case of empty list as shape, since fan_in and fan_out
    # are numerically added below. Without this, GetFanInFanOut() would
    # return None, None instead.
    if isinstance(scale_shape, (list, tuple)) and not scale_shape:
      fan_in, fan_out = 1, 1
    else:
      fan_in, fan_out = GetFanInFanOut(scale_shape)
    if mode == 'fan_in':
      scale_inner = scale / max(1., fan_in)
    elif mode == 'fan_out':
      scale_inner = scale / max(1., fan_out)
    else:
      scale_inner = scale / max(1., (fan_in + fan_out) / 2.)
    if distribution == 'normal' or distribution == 'truncated_normal':
      # constant taken from scipy.stats.truncnorm.std(
      #                         a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale_inner) / .87962566103423978
      return stateless_random_ops.stateless_truncated_normal(
          shape=shape, seed=seed, mean=0.0, stddev=stddev, dtype=dtype)
    elif distribution == 'untruncated_normal':
      stddev = math.sqrt(scale_inner)
      return stateless_random_ops.stateless_random_normal(
          shape=shape, seed=seed, mean=0.0, stddev=stddev, dtype=dtype)
    else:
      limit = math.sqrt(3.0 * scale_inner)
      return stateless_random_ops.stateless_random_uniform(
          shape=shape, seed=seed, minval=-limit, maxval=limit, dtype=dtype)

  return DeterministicVarianceScaling


def _DeterministicRandomXavierUniformInitializer(method, scale, seed):
  """Creates a variance scaling initializer."""

  def XavierUniform(shape, dtype, partition_info):
    """Xavier initialization (x = sqrt(6. / (in + out)); scale*[-x, x])."""
    del partition_info  # Unused.
    if not shape:
      raise ValueError('\'shape\' must not be \'None\' or 0 for XavierUniform')
    fan_in, fan_out = GetFanInFanOut(shape)
    if method == 'xavier':
      limit = math.sqrt(6. / (fan_in + fan_out))
    elif method == 'geo_mean_xavier':
      limit = math.sqrt(3. / math.sqrt(fan_in * fan_out))
    return scale * stateless_random_ops.stateless_random_uniform(
        shape, seed, -limit, limit, dtype)

  return XavierUniform


def _CreateVarInitStateless(name, method, shape, dim0, seed, scale, init_dtype):
  """Creates variable initialization function for a stateless RNG."""
  if (method in [
      'gaussian_sqrt_dim', 'uniform_sqrt_dim', 'truncated_gaussian_sqrt_dim'
  ]):
    if len(shape) > 2:
      # This is probably not the right method to use when len(shape) > 2,
      # e.g. dim0 will be 3 with a 3x3 conv2d kernel.
      tf.logging.warning(
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
  if method in ['gaussian_sqrt_fanavg']:
    fan_in, fan_out = GetFanInFanOut(shape)
    if fan_in is not None and fan_out is not None:
      scale *= math.sqrt(2.0 / (fan_in + fan_out))

  if method in [
      'gaussian', 'gaussian_sqrt_dim', 'gaussian_sqrt_fanin',
      'gaussian_sqrt_fanout', 'gaussian_sqrt_fanavg'
  ]:
    v_init = _DeterministicRandomNormalInitializer(
        seed=seed, mean=0., stddev=scale)
  elif method in ['uniform', 'uniform_sqrt_dim']:
    v_init = _DeterministicRandomUniformInitializer(
        seed=seed, minval=-scale, maxval=scale)
  elif method in ['uniform_positive']:
    v_init = _DeterministicRandomUniformInitializer(
        seed=seed, minval=0., maxval=scale)
  elif method in ['uniform_unit_scaling']:
    v_init = _DeterministicRandomUniformUnitScalingInitializer(
        seed=seed, factor=scale)
  elif method in ['uniform_unit_scaling_fan_avg']:
    v_init = _DeterministicRandomVarianceScalingInitializer(
        scale=scale, mode='fan_avg', distribution='uniform', seed=seed)
  elif method in [
      'truncated_gaussian', 'truncated_gaussian_sqrt_dim',
      'truncated_gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanout'
  ]:
    v_init = _DeterministicRandomTruncatedNormalInitializer(
        seed=seed, mean=0., stddev=scale)
  elif method in ['constant']:
    v_init = init_ops.constant_initializer(value=scale, dtype=init_dtype)
  elif method in ['xavier', 'geo_mean_xavier']:
    v_init = _DeterministicRandomXavierUniformInitializer(method, scale, seed)
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
    v_init = _DeterministicRandomUniformInitializer(
        seed=seed, minval=-bound, maxval=bound)
  else:
    assert False, 'init_type %s not supported.' % method

  return v_init


_global_variable_scope = None


def GetGlobalVariableScope():
  """Gets the global variable scope (as if no variable_scope has been set).

  Returns:
    The VariableScope corresponding to as if no tf.variable_scope is in effect.
  """
  if not _global_variable_scope:
    # Each thread gets its own default global variable scope, and we take
    # advantage of that in order to get a top-level scope. This avoids the
    # need to call tf.get_variable_scope() at the module level, which allows
    # this module to be imported without modifying global state (i.e. creating
    # the default graph). It is important to not mutate the global state at
    # module load time, because it let's us flip flags after import that affect
    # core TensorFlow behavior.
    def Initialize():
      global _global_variable_scope
      _global_variable_scope = tf.get_variable_scope()

    t = threading.Thread(target=Initialize)
    t.start()
    t.join()
  return _global_variable_scope


_GLOBAL_STEP_STACK = ThreadLocalStack()


@contextlib.contextmanager
def GlobalStepContext(global_step_tensor):
  _GLOBAL_STEP_STACK.stack.append(global_step_tensor)
  try:
    yield
  finally:
    _GLOBAL_STEP_STACK.stack.pop()


def GetGlobalStep():
  """Return the global_step."""
  if _GLOBAL_STEP_STACK.stack:
    return _GLOBAL_STEP_STACK.stack[-1]
  return tf.train.get_global_step()


def GetOrCreateGlobalStepVar():
  """Return the global_step variable, creating it if it does not exist.

  Prefer GetGlobalStep if a tensor rather than a tf.Variable is sufficient.

  Returns:
    The global_step variable, or a new created one if it does not exist.
  """
  with tf.variable_scope(GetGlobalVariableScope(), use_resource=True):
    if _FromGlobal('pin_vars_to_cpu'):
      with tf.device('/cpu:0'):
        return tf.train.get_or_create_global_step()
    else:
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

  class AddIdentity:
    """Helper class."""

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


def _GetVarsToLoad(all_vars, variable_loading_rules, var_ignore_rules,
                   ckpt_path):
  """Determines variables to load and their names in checkpoint."""
  # This list contains mappings from var names as they appear in the checkpoint
  # to the vars in our model they correspond to.
  vars_to_load = []
  for model_var in all_vars:
    loaded = False
    for regexp, name_format in variable_loading_rules:
      match = re.match(regexp, model_var.name)
      # Skip if var doesn't match the loading rules, or if it should be ignored.
      if not match:
        tf.logging.debug('Loading rules do not match %s.', model_var.name)
        continue
      elif any(re.match(r, model_var.name) for r in var_ignore_rules):
        tf.logging.debug('Ignoring %s from loading.', model_var.name)
        continue
      checkpoint_var_name = name_format % match.groups()
      if checkpoint_var_name.endswith(':0'):
        checkpoint_var_name = checkpoint_var_name[:-2]
      tf.logging.info('Loading %s from %s with regexp: %s', model_var.name,
                      checkpoint_var_name, regexp)
      vars_to_load.append((checkpoint_var_name, model_var))
      loaded = True
      break
    if not loaded:
      tf.logging.info(
          'Not loading model variable %s from %s as it does not match any rules'
          ' or matches ignored', model_var.name, ckpt_path)
  return vars_to_load


def OverrideVarsFromCheckpoint(all_vars, checkpoint_path,
                               variable_loading_rules, var_ignore_rules):
  """Add TF graph ops to override variables from a provided checkpoint.

  Args:
    all_vars: List of all the parameters in the model.
    checkpoint_path: A path to the checkpoints of a pretrained model.
    variable_loading_rules: A list of tuples of strings defining (regex to match
      parameter names in the model to override, format string to determine the
      corresponding var in the checkpoint).
    var_ignore_rules: A list consisting of a list of regexes to match parameter
      names in the model which should not be overridden, even if they match
      those in the loading rules.

  Returns:
    A callable that, when called with a tf.Session, will restore the variables
    from the provided checkpoint.
  """
  vars_to_load = _GetVarsToLoad(all_vars, variable_loading_rules,
                                var_ignore_rules, checkpoint_path)
  if not vars_to_load:
    all_rules_text = '\n'.join(
        [f'{k} --> {v}' for k, v in variable_loading_rules])
    raise ValueError(f'Variable loading rules {all_rules_text} '
                     f'did not match any of {len(all_vars)} vars.')
  load_var_names = '\n'.join(sorted([v.name for _, v in vars_to_load]))
  tf.logging.info(f'Overriding {len(vars_to_load)} vars from '
                  f'{checkpoint_path}:\n{load_var_names}')

  savers = []
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
    savers.append(tf.train.Saver(var_list=unique_vars_to_load, sharded=True))
    vars_to_load = remaining_vars_to_load

  def _Restore(sess):
    for saver in savers:
      saver.restore(sess, checkpoint_path)

  return _Restore


def OverrideVarsFromCheckpoints(all_vars, ckpts_loading_rules):
  """Add TF graph ops to override model variables from checkpoints.

  Args:
    all_vars: List of all the parameters in the model.
    ckpts_loading_rules: A dictionary of checkpoint path: loading rules.
      Checkpoint path must be a path to a pretrained model, and loading rules is
      expected to be a tuple of two lists. The first consisting of tuples of
      strings defining (regex to match parameter names in the model to override,
      format string to determine the corresponding var in the checkpoint), and
      the second list consisting of a list of regexes to match parameter names
      in the model which should not be overridden, even if they match those in
      the loading rules.

  Returns:
    A callable that, when called with a tf.Session, will restore the variables
    from checkpoint and return a list of overwritten variables.

  Raises:
    ValueError: if colliding vars exist or loading rules is not a list.
  """
  if len(ckpts_loading_rules) > 1:
    tf.logging.info('Overriding vars from multiple checkpoints.')

  var_refs_overridden = set()
  var_names_overridden = set()
  restore_fns = []
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
    to_load_vars = _GetVarsToLoad(all_vars, loading_rules[0], loading_rules[1],
                                  ckpt_path)
    var_refs_to_override = [var[1].experimental_ref() for var in to_load_vars]
    var_names_to_override = [var[1].name for var in to_load_vars]

    overlap_refs = set.intersection(var_refs_overridden, var_refs_to_override)
    if overlap_refs:
      raise ValueError('Colliding variables to override: %s' % overlap_refs)

    restore_fns.append(
        OverrideVarsFromCheckpoint(all_vars, ckpt_path, loading_rules[0],
                                   loading_rules[1]))
    var_refs_overridden.update(var_refs_to_override)
    var_names_overridden.update(var_names_to_override)
  tf.logging.info('Model variables overridden: %s', var_refs_overridden)

  def _Restore(sess):
    for fn in restore_fns:
      fn(sess)
    return var_names_overridden

  return _Restore


def ComputeGradientsSimple(loss_or_activations,
                           all_vars,
                           grad_aggregation_method,
                           colocate_gradients_with_ops,
                           gate_gradients,
                           activations_grad=None):
  return tf.gradients(
      loss_or_activations,
      all_vars,
      grad_ys=activations_grad,
      aggregation_method=grad_aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      gate_gradients=gate_gradients)


def ComputeTpuEmbeddingGradients(task_name, loss, activation_dict,
                                 tpu_embedding_collection):
  """Returns a TpuEmbedding SendGradient op.

  Args:
    task_name: The name of the task to compute the gradients for.
    loss: The loss to backprop from.
    activation_dict: String feature -> embedding activations dict.
    tpu_embedding_collection: TpuEmbeddingCollection instance.

  Returns:
    A tuple (send_gradient_op, eval_metrics):
    - send_gradient_op: A TF op that sends gradients to TPU Embedding.
    - eval_metrics: A list of (name, value, weight) tuples containing the eval
      metrics, where value and weight are scalar tensors.
  """
  # Scale the loss to account for the full batch size.
  shards = tpu_function.get_tpu_context().number_of_shards
  loss *= tf.constant(1.0 / shards, dtype=loss.dtype)
  gradients = tf.gradients(
      loss,
      list(activation_dict.values()),
      # TPUEmbedding expects gradients for all embedding
      # features, so return zero if unconnected.
      unconnected_gradients=tf.UnconnectedGradients.ZERO)

  # Apply gradient multiplier schedule.
  grad_multiplier = (
      tpu_embedding_collection.gradient_multiplier_schedule.Value())
  gradients = [g * grad_multiplier for g in gradients]

  feature_to_gradient_dict = py_collections.OrderedDict(
      zip(list(activation_dict.keys()), gradients))
  send_gradient_op = (
      tpu_embedding_collection.tpu_embedding.generate_send_gradients_op(
          feature_to_gradient_dict, step=GetGlobalStep()))

  # Add eval metrics. Note we can't add these to tpu_embedding_collection since
  # it'll leak these to a different graph from a different task in multi-task
  # setup.
  # TODO(laigd): find a better solution using tpu_embedding_collection.
  eval_metrics = [
      (f'tpu_embedding_activation_norm/{task_name}/all',
       tf.sqrt(SumSquared(activation_dict.values())), tf.constant(1.0)),
      (f'tpu_embedding_grad_norm/{task_name}/all',
       tf.sqrt(SumSquared(gradients)), tf.constant(1.0)),
      ('tpu_embedding_gradient_multiplier', grad_multiplier, tf.constant(1.0)),
  ]
  return send_gradient_op, eval_metrics


def _ComputeGradientsTpu(loss_or_activations,
                         all_vars,
                         grad_aggregation_method,
                         colocate_gradients_with_ops,
                         gate_gradients,
                         skip_zero_gradients=None,
                         use_bf16_gradients_ar=False,
                         defer_crs_to_apply_grad=False,
                         activations_grad=None,
                         is_activations=False):
  """Computes gradients for local loss across whole TPU cluster.

  This implementation specializes for the case where weight params maybe used
  for different number of times in the forward computation, so that gradients
  should be normalized by the actual number of times they are being computed.

  TODO(yonghui): Maybe merge this implementation with the _ComputeGradientsTpu
  one.

  Args:
    loss_or_activations: The loss or activations to backprop from.
    all_vars: Vars with respect to which gradients are to be computed.
    grad_aggregation_method: aggregation method to use when calling
      tf.gradients.
    colocate_gradients_with_ops: boolean, whether or not to colocate gradient op
      with the original op.
    gate_gradients: boolean, flag to be passed to tf.gradients.
    skip_zero_gradients: whether to skip zero gradients during aggregation.
    use_bf16_gradients_ar: Whether to use bfloat16 dtype for gradients
      all-reduce.
    defer_crs_to_apply_grad: Whether to defer gradient cross replica sum to
      apply_gradient. This helps reducing the number of gradient all-reduces
      when doing gradient accumulation, which does gradient cross replica sum
      only every k steps in a tf.cond. Currently this works only when
      skip_zero_gradients is None.
    activations_grad: The gradients computed for activations.
    is_activations: A boolean, whether the input is loss or activations.

  Returns:
    Gradients to be passed back.

  Raises:
    ValueError: upon invalid arguments.
  """
  if is_activations:
    assert activations_grad is not None

  if not skip_zero_gradients and not is_activations:
    # Scale the loss to account for the full batch size.
    shards = tpu_function.get_tpu_context().number_of_shards
    assert shards
    loss_or_activations *= tf.constant(
        1.0 / shards, dtype=loss_or_activations.dtype)

  # Computes the gradients.
  # Sum the grads so that we can compute statistics across the whole batch.
  all_grads = ComputeGradientsSimple(
      loss_or_activations=loss_or_activations,
      all_vars=all_vars,
      grad_aggregation_method=grad_aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops,
      gate_gradients=gate_gradients,
      activations_grad=activations_grad)

  # NOTE: We can't use tpu_optimizer.CrossShardOptimizer since
  # we need to scale the grads *after* the cross_replica_sum to
  # match GPU version!

  # TODO(cwhipkey): should we do something different here? - we could do
  # some operations on the gradients before the aggregation (see comments in
  # tensorflow/contrib/tpu/python/tpu/tpu_optimizer.py - see compute_gradients -
  # for some more details).

  aggregated_grads = []
  for g in all_grads:
    if g is None:
      aggregated_grads.append(None)
      continue
    if use_bf16_gradients_ar:
      g = tf.cast(g, tf.bfloat16)
    with tf.ops.colocate_with(g):
      if skip_zero_gradients is None:
        # loss is already scaled by 1/shards.
        if defer_crs_to_apply_grad:
          normalized_g = tf.convert_to_tensor(g)
        else:
          normalized_g = tf.tpu.cross_replica_sum(g)
      else:
        # Compute the cross-replica mean of 'g', skipping zero gradients.

        # Q(yonghui): Is there a better way to detect a non-zero gradient?
        # Note(yonghui): gradient of a weight can be zero if that
        # weight is not used in the forward computation, e.g. as in
        # switchable layers in neural architecture search, pruned by channel
        # mask, or sparsified.
        if skip_zero_gradients == 'weight':
          # Same shape as 'g'.
          g_is_non_zero = tf.cast(tf.math.abs(g) > 1e-8, g.dtype)
        elif skip_zero_gradients == 'variable':
          # A variable-wide 0/1 scalar.
          g_is_non_zero = tf.cast(
              tf.reduce_sum(tf.math.abs(g)) > 1e-24, g.dtype)
        else:
          raise ValueError('Unknown skip_zero_gradients: %s' %
                           skip_zero_gradients)
        num_updates = tf.maximum(tf.tpu.cross_replica_sum(g_is_non_zero), 1.0)
        normalized_g = tf.tpu.cross_replica_sum(g) / num_updates
      aggregated_grads.append(normalized_g)
  return aggregated_grads


class VarGrad:
  """A class that holds a variable and a gradient."""

  _VAR_GRAD = py_collections.namedtuple('VarGradNamedTuple', ['var', 'grad'])

  def __init__(self, *args, **kwargs):
    self._var_grad = self._VAR_GRAD(*args, **kwargs)

  def __getitem__(self, key):
    return self._var_grad[key]

  def __getattr__(self, key):
    return getattr(self._var_grad, key)

  def __iter__(self):
    return iter(self._var_grad)

  def __repr__(self):
    return 'VarGrad(%r, %r)' % (self._var_grad.var, self._var_grad.grad)


def SkipNoneGradients(var_grads):
  """Removes pairs whose grad is None."""
  for key, (_, g) in var_grads.FlattenItems():
    if g is None:
      tf.logging.info('ComputeGradients drops %s', key)
  return var_grads.Filter(lambda var_grad: var_grad.grad is not None)


def ComputeGradients(
    loss_or_activations,
    vmap,
    grad_aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
    colocate_gradients_with_ops=True,
    gate_gradients=False,
    compute_gradients_fn=None,
    skip_zero_gradients=None,
    use_bf16_gradients_ar=False,
    skip_none_gradients=True,
    defer_crs_to_apply_grad=False,
    activations_grad=None,
    is_activations=False):
  """Computes gradients of variables in vmap w.r.t loss.

  Args:
    loss_or_activations: either the loss, which is a scalar tensor, or
      activations, which could be a tensor or a list of tensors.
    vmap: A `.NestedMap` of variables.
    grad_aggregation_method: Specifies the method used to combine gradient
      terms. Accepted values are constants defined in the class
      AggregationMethod.
    colocate_gradients_with_ops: If True, try colocating gradients with the
      corresponding op.
    gate_gradients: If True, add a tuple around the gradients returned for an
      operations. This avoids some race conditions.
    compute_gradients_fn: Function to use to compute gradients. If None, use
      default. compute_gradients_fn should have the same signature as this
      function, but without the last argument.
    skip_zero_gradients: Whether to skip aggregating zero gradients. This helps
      in case where some weights may not be used in forward computation, e.g.,
      sparsely activated networks or switchable layers in neural architectural
      search. Only applicable on TPU.
      Possible values are:

        - None: do not skip zero gradients;
        - `variable`: skip if the entire variable's gradients are almost zero;
          reduce_sum(abs(grads)) < 1e-8.
        - `weight`: skip if the individual weight's gradients are almost zero:
          abs(grad) < 1e-8.

    use_bf16_gradients_ar: Whether to use bfloat16 dtype for gradients
      all-reduce. This applies to TPU only.
    skip_none_gradients: Whether to skip gradients that are None.
    defer_crs_to_apply_grad: Whether to defer gradient cross replica sum to
      apply_gradient. This applies to TPU only.
    activations_grad: The gradients computed for activations.
    is_activations: A boolean, whether the input is loss or activations.

  Returns:
    var_grad - a `.NestedMap` of VarGrad. You can view
    var_grad as an ordered list of (key, (var, grad)) tuples. Every
    key of var_grad exists in vmap. Every variable in vmap that
    contributes to loss must exist in var_grad. Every var of var_grad
    must exist in vmap.  grad is the corresponding gradient computed
    for var. grad is guaranteed to be not None.
  """
  if not is_activations:
    loss_or_activations = HasRank(loss_or_activations, 0)
  assert isinstance(vmap, NestedMap)
  assert skip_zero_gradients in (None, 'variable', 'weight')

  # Uniqify and remove None.
  filtered_vmap = vmap.Filter(_Unique())
  assert filtered_vmap is not None

  # Filter out variables not contributing to 'loss_or_activations'.
  # This doesn't work if the training loop is wrapped inside a tf.function,
  # since all variables will be lifted out and trainable_variables will be
  # empty. In that case we skip the check.
  trainable_variables = set(tf.trainable_variables())
  if trainable_variables:

    def Needed(v):
      if isinstance(v, tf.Variable):
        if v not in trainable_variables:
          # Skip non-trainable variables. Otherwise,
          # tf.Optimizer.apply_gradients throws up an exception instead
          # of skipping the update.
          return False
      return True

    filtered_vmap = filtered_vmap.Filter(Needed)
    assert filtered_vmap is not None

  filtered_vlist = filtered_vmap.Flatten()

  # Use caller-supplied gradient function if supplied.
  if compute_gradients_fn is not None:
    take_grad = compute_gradients_fn
  else:
    # tpu vs non-tpu is slightly different.
    if use_tpu():
      take_grad = functools.partial(
          _ComputeGradientsTpu,
          skip_zero_gradients=skip_zero_gradients,
          use_bf16_gradients_ar=use_bf16_gradients_ar,
          defer_crs_to_apply_grad=defer_crs_to_apply_grad,
          activations_grad=activations_grad,
          is_activations=is_activations)
    else:
      take_grad = ComputeGradientsSimple

  grads = take_grad(loss_or_activations, filtered_vlist,
                    grad_aggregation_method, colocate_gradients_with_ops,
                    gate_gradients)

  # Formulate pairs of (var, grad) and pack them into the same
  # structure as filtered_vmap.
  var_grads = filtered_vmap.Pack(
      [VarGrad(v, g) for v, g in zip(filtered_vlist, grads)])

  if skip_none_gradients:
    var_grads = SkipNoneGradients(var_grads)

  return var_grads


def MaskGradients(var_grad, grad_mask):
  """Computes gradients of non-masked variables in vmap w.r.t loss.

  Args:
    var_grad: A `.NestedMap` of (variable, gradient)
    grad_mask: A dict of (variable name, mask).

  Returns:
    var_grad - a `.NestedMap` of (variable, mask * gradient).
  """

  def ApplyMask(entry):
    var, grad = entry
    mask = grad_mask[var.name]
    if isinstance(grad, tf.IndexedSlices):
      return VarGrad(var, tf.IndexedSlices(grad.values * mask, grad.indices))
    else:
      return VarGrad(var, grad * mask)

  return var_grad.Transform(ApplyMask)


def ApplyGradMultiplier(vs_gs, grad_scale=None):
  """Scale gradients by grad_scale on same device as corresponding variables.

  Args:
    vs_gs: A `.NestedMap` of VarGrad.
    grad_scale: If None, each vs_gs entry has the scale. Otherwise, grad_scale
      applies to every entry.

  Returns:
    A `.NestedMap` of (variable, gradient * grad_scale). In particular, if
    grad_scale is 0, the result gradient is always 0, even if the input
    gradient is inf or nan.
  """

  def ScaleOrZero(var, grad, scale):
    grad = CheckNumerics(grad, 'Gradient for %s is not finite.' % var.name)
    return tf.where(
        tf.equal(scale, 0.), tf.zeros_like(grad),
        tf.cast(scale, grad.dtype) * grad)

  def Scale(item):
    """Scales the gradient."""
    var, grad = item
    assert grad is not None, ('No grad found for ', var.name)
    if grad_scale is None:
      scale = item.scale
    else:
      scale = grad_scale
    with tf.device(var.device):
      if isinstance(grad, tf.IndexedSlices):
        grad = tf.IndexedSlices(
            ScaleOrZero(var, grad.values, scale), grad.indices,
            grad.dense_shape)
      else:
        grad = ScaleOrZero(var, grad, scale)
    return VarGrad(var, grad)

  return vs_gs.Transform(Scale)


def HasNanOrInf(x):
  if isinstance(x, tf.IndexedSlices):
    x = x.values
  with tf.device(x.device):
    if x.dtype.is_complex:
      return tf.reduce_any(
          [HasNanOrInf(tf.math.real(x)),
           HasNanOrInf(tf.math.imag(x))])
    return tf.reduce_any(
        tf.math.logical_or(tf.math.is_nan(x), tf.math.is_inf(x)))


def HasNanOrInfGradient(var_grads):
  """Returns a bool tensor to indicate if `var_grads` contains NaNs or Infs.

  Args:
    var_grads: A `.NestedMap` with (var, grad) tuple as the map value.

  Returns:
    A bool scalar tensor to indicate if the `var_grads` contains NaNs or Infs.
  """
  return tf.reduce_any([HasNanOrInf(g) for (_, g) in var_grads.Flatten()])


def ApplyGradNormClipping(vs_gs, norm=1.0):
  """Clip gradients to norm on same device as corresponding variables.

  Args:
    vs_gs: A `.NestedMap` of VarGrad.
    norm: Each tensor's gradient will be scaled down to have a maximum L2-norm
      value of `norm`.

  Returns:
    A `.NestedMap` of VarGrad(variable, scaled_gradient). In particular, if
    grad_scale is 0, the result gradient is always 0, even if the input
    gradient is inf or nan.
  """

  def ClipByNorm(var, grad, norm):
    grad = CheckNumerics(grad, 'Gradient for %s is not finite.' % var.name)
    return tf.clip_by_norm(grad, norm)

  def Clip(item):
    """Scales the gradient."""
    var, grad = item
    assert grad is not None, ('No grad found for ', var.name)
    with tf.device(var.device):
      if isinstance(grad, tf.IndexedSlices):
        grad = tf.IndexedSlices(
            ClipByNorm(var, grad.values, norm), grad.indices, grad.dense_shape)
      else:
        grad = ClipByNorm(var, grad, norm)
    return VarGrad(var, grad)

  return vs_gs.Transform(Clip)


SKIP_LP_REGULARIZATION = '__lingvo_skip_lp_regularization'


def AdjustGradientsWithLpLoss(var_grads, lp_regularizer_weight, p=2.0):
  """Adjusts the map of (var, grad) with Lp regularization, where p=1.0 or 2.0.

  Args:
    var_grads: a `.NestedMap` or list of (variable, gradient).
    lp_regularizer_weight: Lp regularization weight.
    p: For now we support 1.0 or 2.0.

  Returns:
    A tuple (lp_loss, var_grads).

    - lp_loss: A scalar. The lp loss.
    - var_grads: a `.NestedMap` or list of (variable, gradient) regulated by Lp.
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

  def ShouldAdjust(v):
    return v not in tf.get_collection(SKIP_LP_REGULARIZATION)

  filtered_var_grads = [
      var_grad for var_grad in Flatten(var_grads) if ShouldAdjust(var_grad.var)
  ]
  filtered_vars = Transform(GetVar, filtered_var_grads)
  for v in filtered_vars:
    tf.logging.info('AdjustGradientsWithLpLoss: %s', v.name)

  if p == 2.0:
    lp_loss = 0.5 * lp_regularizer_weight * SumSquared(filtered_vars)
  elif p == 1.0:
    lp_loss = lp_regularizer_weight * SumAbs(filtered_vars)

  def LpGrad(var_grad):
    """Adjusts item's grad w/ Lp loss term."""
    var, grad = var_grad
    if isinstance(grad, tf.IndexedSlices):
      # Question(rpang): do we apply Lp loss here even if 'var' is in
      # SKIP_LP_REGULARIZATION?
      #
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
        # occurrences in 'ids'.
        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(ids, dtype=values.dtype), ids, vocab_size)

        # Gradients for duplicated ids will be summed when they get
        # applied, and hence we account for that by first dividing
        # gradient resulting from lp loss by how many times the id is
        # duplicated.
        #
        # For each id in 'ids', we know counts[id] is non-zero,
        # hence, it's always safe to take reciprocal.
        weights = tf.math.reciprocal(tf.gather(counts, ids))
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
    return VarGrad(var, grad)

  return lp_loss, Transform(LpGrad, var_grads)


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
    for key, val in x.items():
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


def WeightedAvg(values, weights, sum_reduction_fn=tf.reduce_sum, name=''):
  """Computes weighted average of values from a tensor.

  Args:
    values: a tensor of values
    weights: a tensor of weights
    sum_reduction_fn: called to reduce the values and weights to single value
    name: name of metric.

  Returns:
    A tuple (avg, total_weight).

    - avg: weighted average value
    - total_weight: sum of all weights
  """
  msg = 'shape of values and weights tensors must match for metric ' + name
  values = with_dependencies(
      [assert_equal(tf.shape(values), tf.shape(weights), message=msg)], values)
  total_weight = sum_reduction_fn(weights)
  # divide_no_nan only supports tf.{float,complex}*.
  dtype = values.dtype if values.dtype is tf.float64 else tf.float32
  avg = tf.math.divide_no_nan(
      sum_reduction_fn(tf.cast(values, dtype) * tf.cast(weights, dtype)),
      tf.cast(total_weight, dtype))
  return tf.cast(avg, values.dtype), total_weight


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
    for name, (value, weight) in m.items():
      if name not in lists_of_metrics:
        lists_of_metrics[name] = []
      lists_of_metrics[name].append((value, weight))

  for name, values_and_weights in sorted(lists_of_metrics.items()):
    values = tf.stack([x[0] for x in values_and_weights])
    weights = tf.stack([x[1] for x in values_and_weights])
    ret_dict[name] = WeightedAvg(values, weights, tf.reduce_sum, name)

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
    for name, value in m.items():
      if name not in lists_of_per_example:
        lists_of_per_example[name] = []
      lists_of_per_example[name].append(value)

  for name, values in sorted(lists_of_per_example.items()):
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
  all_keys = set(
      [k for loss_metrics, _ in loss_metric_weight_pairs for k in loss_metrics])  # pylint: disable=g-complex-comprehension
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


def AddVN(p, x, per_step=False):
  """Add variational noise to x.

  Args:
    p: Layer params, with a `vn` subparam containing `VariationalNoiseParams`.
    x: Input to add variational noise to.
    per_step: Whether to add per_step noise.

  Returns:
    The input with variational noise added according to params.
  """
  if per_step:
    if not p.vn.per_step_vn:
      return x
  else:
    if not p.vn.global_vn:
      return x

  if p.vn.scale is None:
    raise ValueError('VN scale must be set.')

  if p.vn.deterministic:
    seeds = GenerateStepSeedPair(p, GetGlobalStep())
    if not p.vn.per_step_vn:
      # First element of seeds is global step.
      seeds = tf.stack([tf.zeros_like(seeds[0]), seeds[1]])
    noises = DeterministicVN(p, seeds, tf.shape(x), mean=0.0, std=1.0)
  else:
    seed = p.vn.seed
    if seed and p.vn.per_step_vn:
      # TODO(b/171767456): Fix per_step_vn.
      # seed += GetGlobalStep() * 203984
      pass
    noises = tf.random.normal(tf.shape(x), stddev=1.0, seed=seed, dtype=x.dtype)
  noises = tf.cast(p.vn.scale, x.dtype) * noises
  return x + noises


def VariationalNoiseParams(scale,
                           global_vn=False,
                           per_step_vn=False,
                           seed=None,
                           deterministic=False):
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
  p.Define(
      'deterministic', deterministic, 'If true, generate noise using'
      'stateless random ops that are compatible with TF functional ops.')
  return p


def DefaultVN():
  return VariationalNoiseParams(
      scale=None,
      global_vn=False,
      per_step_vn=False,
      seed=None,
      deterministic=False)


# To disable VN of a layer, we use 1.0 in the first input parameter
# of the following function because otherwise it is the same to DefaultVN()
# which will be updated by parent configuration in CopyBaseParams()
def DisableVN():
  return VariationalNoiseParams(1.0, False, False)


# Step seed keyed by graph.
_STEP_SEED_DICT = ThreadLocalDict()


def GetStepSeed():
  """Gets step_seed."""
  key = id(tf.get_default_graph())
  if key not in _STEP_SEED_DICT.dict:
    ResetStepSeed()
  return _STEP_SEED_DICT.dict[key]


def ResetStepSeed(seed=0):
  """Resets step_seed to specified value."""
  key = id(tf.get_default_graph())
  _STEP_SEED_DICT.dict[key] = tf.convert_to_tensor(seed, dtype=tf.int64)


def MaybeResetStepSeedFromScope():
  """In graph mode, resets step_seed according to the current named scope.

  This is used in graph mode to avoid "tensor is from a different graph"
  errors that happen when we share random seend tensors too much.
  See b/129159299 for more context.

  Eager mode does not have this problem, so in eager mode we do nothing.
  """
  if not tf.executing_eagerly():
    ResetStepSeed(GenerateSeedFromName(tf.no_op(name='new_step_seed').name))


def MaybeResetStepSeed(seed):
  """If we're in graph mode, reset the step seed."""
  if not tf.executing_eagerly():
    ResetStepSeed(seed)


def GetIncStepSeed():
  """Returns and increments the step_seed."""
  step_seed = GetStepSeed()
  # TODO(lepikhin): introduce a routine filling a queue of uint32 random seeds
  # independent of underlying PRNG used by tensorflow.
  ResetStepSeed(step_seed + 1)
  return step_seed


def GenerateStepSeedPair(p, op_seed=None):
  """Generates a seed pair for deterministic random operations in ...

  functional loops.

  This function retrieves a unique seed pair on each call, based off the current
  global step and step seed. The step seed ensures this function returns a
  unique seed pair on each call: calling this function automatically increments
  the step seed. The step seed is automatically reset at the beginning of each
  global step in the model's FProp and works transparently through recurrent.py.

  Args:
    p: A hyperparams.Params object, containing keys 'random_seed' and
      'is_inference'.
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
    return tf.random.uniform([2], maxval=seed_dtype.max, dtype=seed_dtype)

  global_step = tf.cast(GetGlobalStep(), seed_dtype)
  step_seed = tf.cast(GetIncStepSeed(), seed_dtype)
  seeds = tf.stack([global_step, step_seed])

  if p.random_seed is not None:
    seeds += p.random_seed
  if op_seed is not None:
    op_seed = tf.cast(op_seed, seed_dtype)
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
    random_tensor = keep_prob + tf.random.stateless_uniform(
        noise_shape, seed=seeds, dtype=tf.float32)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    if x.dtype != tf.float32:
      binary_tensor = tf.cast(binary_tensor, x.dtype)
      keep_prob = tf.cast(keep_prob, dtype=x.dtype)
    result = tf.div(x, keep_prob) * binary_tensor
    result.set_shape(x.get_shape())
    return result


def DeterministicVN(params, seeds, noise_shape, mean=0.0, std=1.0, name=None):
  """Produces Fully deterministic Gaussian noise from shape, mean and std.

  Args:
    params: Nested map of params.
    seeds: A Tensor of shape [2]. 2 seeds for deterministic random number
      generator.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the shape for
      randomly generated Gaussian noise.
    mean: Mean for the Gaussian noise.
    std: Standard deviation for noise.
    name: An optional name for this operation.

  Returns:
    A Tensor with the shape noise_shape and type fprop_dtype.
  """

  with tf.name_scope(name, 'gaussian_noise') as name:
    if use_tpu():
      seeds = tf.cast(seeds, tf.int32)
    random_tensor = mean + (
        std * tf.random.stateless_normal(noise_shape, seed=seeds))
    if FPropDtype(params) != tf.float32:
      random_tensor = tf.cast(random_tensor, FPropDtype(params))
    return random_tensor


BATCH_NORM_UPDATES = 'batch_norm_updates'

_BATCH_NORM_UPDATES_DICT = '__batch_norm_update_dict'
_get_batch_norm_updates_dict = _CollectionGetter(_BATCH_NORM_UPDATES_DICT,
                                                 lambda: {})


def UpdateBatchNormVars(batch_norm_var, batch_norm_stats, decay):
  """Update batch normalization moving averages."""
  with tf.name_scope(
      'AssignMovingAvg', values=[
          batch_norm_var,
          batch_norm_stats,
          decay,
      ]) as scope:
    with tf.ops.colocate_with(batch_norm_var):
      decay = tf.convert_to_tensor(
          1.0 - decay, dtype=batch_norm_var.dtype.base_dtype)
      update_delta = (batch_norm_var - tf.cast(
          batch_norm_stats, batch_norm_var.dtype.base_dtype)) * decay
      has_nan_or_inf = tf.reduce_any(
          tf.math.logical_or(
              tf.math.is_nan(update_delta), tf.math.is_inf(update_delta)))
      update_delta = tf.cond(has_nan_or_inf,
                             lambda: tf.zeros_like(update_delta),
                             lambda: update_delta)
      bn_update = tf.assign_sub(batch_norm_var, update_delta, name=scope)
  tf.add_to_collection(BATCH_NORM_UPDATES, bn_update)
  if not tf.executing_eagerly():
    bn_update_dict = _get_batch_norm_updates_dict()
    assert bn_update.name not in bn_update_dict
    bn_update_dict[bn_update.name] = (batch_norm_var, batch_norm_stats)
  return bn_update


def FindRelevantBatchNormUpdates(loss, batch_norm_updates):
  """Finds and returns a list of relevant batch-normalization updates.

  Args:
    loss: The loss that is being optimized for. A tensor or a list of tensors.
    batch_norm_updates: A list of batch normalization updates.

  Returns:
    A pair of lists. The first list contains all the batch normalization updates
    that are relevant to the loss being optimized, and the second list contains
    all in batch_norm_updates but not in the first list.
  """
  if tf.executing_eagerly():
    return [], []
  dependent_ops_and_tensors = set(FindNeeded(loss))
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


_SAMPLE_STEP_STACK = ThreadLocalStack()


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
  try:
    _SAMPLE_STEP_STACK.stack.append(step)
    yield step
  finally:
    _SAMPLE_STEP_STACK.stack.pop()


def _GetSampleStep():
  return _SAMPLE_STEP_STACK.stack[-1] if _SAMPLE_STEP_STACK.stack else None


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
  if _FromGlobal('print_debug_tensors'):
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


def clip_by_value(t, clip_value_min, clip_value_max, name=None):  # pylint: disable=invalid-name
  if t.dtype.is_complex:
    return tf.complex(
        tf.clip_by_value(
            tf.math.real(t), clip_value_min, clip_value_max, '%s_real' % name),
        tf.clip_by_value(
            tf.math.imag(t), clip_value_min, clip_value_max, '%s_imag' % name))
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


def ReduceRms(x: tf.Tensor) -> tf.Tensor:
  """Computes root mean square of tensor x with numerical stability."""
  if not x.shape.is_fully_defined():
    raise ValueError('Shape of x must be fully defined.')

  if not x.shape.as_list():
    return x

  denom = functools.reduce((lambda x, y: x * y), x.shape.as_list())
  if denom <= 1e8:
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(x)))

  tf.logging.info('reduce_rms %s denom=%d', x, denom)
  sum_square_x = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.square(x), -1))
  avg_square_x = sum_square_x / tf.constant(denom, dtype=sum_square_x.dtype)
  return tf.math.sqrt(avg_square_x)


def PiecewiseConstant(x_in, boundaries, values, vdtype):
  """Returns the piecewise value of x_in."""
  x_in = tf.cast(tf.convert_to_tensor(x_in), tf.float32)
  assert len(values) == len(boundaries) + 1
  assert sorted(boundaries) == list(boundaries)
  bs = tf.convert_to_tensor(boundaries, dtype=tf.float32)
  vs = tf.convert_to_tensor(values, dtype=vdtype)
  # The following is equivalent to 'return vs[index]'.
  index = tf.reduce_sum(tf.cast(tf.greater_equal(x_in, bs), tf.int32))
  one_hot_vec = tf.one_hot(
      tf.expand_dims(index, 0), depth=len(values), dtype=vdtype)
  return Matmul(tf.reshape(vs, (1, -1)), tf.transpose(one_hot_vec))[0][0]


def PadSequenceDimension(x, length, pad_val, shape=None, axis=1):
  """Pads x to `length` using `pad_val` along the axis dim.

  Assumes `x` is a tensor with rank >= 2, and it only pads `x` to `length`
  along the axis dim. Explicitly sets the returned tensor shape to `shape` if
  given. Raises runtime errors if x.shape[axis] > length or
  x.shape[i] != shape[i] where i != axis.

  Args:
    x: the tensor to be padded with axis dimension being the time. E.g., x
      usually has shape [batch, seq_len, ...], when axis=1.
    length: an int to specify the length to pad x to.
    pad_val: an int or float used to pad x.
    shape: an int array specifying the shape of the padded tensor if specified.
    axis: The dimension that x will be padded, default to 1.

  Returns:
    The padded tensor with shape [batch, seq_len, ...], where
    ret[:, :seq_len, ...] == x, when axis=1, and similarly for other axes.
  """
  if x.shape.ndims is not None:
    rank = x.shape.ndims
    assert rank >= 2
    slen = GetShape(x, rank)[axis]
    pad_len = length - slen
    pad = [[0, 0] for _ in range(rank)]
    pad[axis][1] = pad_len
  else:
    rank = tf.rank(x)
    with tf.control_dependencies([assert_greater_equal(rank, 2)]):
      slen = tf.shape(x)[axis]
    pad_len = length - slen
    pad = tf.scatter_nd([[axis, 1]], [pad_len], [rank, 2])
  x = tf.pad(x, pad, constant_values=pad_val)
  if x.shape.ndims is not None and isinstance(length, int):
    static_shape = x.shape.as_list()
    static_shape[axis] = length
    x.set_shape(static_shape)

  if shape:
    if not isinstance(shape, (list, tuple)):
      raise TypeError('Shape must be a list or tuple.')
    x = HasRank(x, len(shape))
    x = tf.ensure_shape(x, shape)
  return x


def PadSequenceTo(xs, padding, length, pad_val):
  """Pads `xs` and `padding` to `length` using `pad_val` along the 2nd dim.

  Pads `xs` to `length` using `pad_val`, and `padding` using 1.
  Raise error if `x.shape[:2]` and `padding.shape` are not the same.

  Args:
    xs: A Tensor or a list of Tensors of shape [batch, seqlen] or [batch,
      seqlen, ...].
    padding: A 0/1 Tensor of shape [batch, seqlen]. 1 is for padded locations.
    length: A Python int, the length to pad to.
    pad_val: A Python numeric, used for padding x.

  Returns:
    A tuple of padded xs and padding.
  """
  if not isinstance(xs, (list, tuple)):
    new_xs = [xs]
  else:
    new_xs = xs

  res = []
  for x in new_xs:
    batch, slen = GetShape(x, 2)

    padding = HasRank(padding, 2)
    padding = HasShape(padding, [batch, slen])

    new_x = PadSequenceDimension(x, length, pad_val)
    res.append(new_x)
  padding = PadSequenceDimension(padding, length, tf.cast(1, padding.dtype))

  if not isinstance(xs, (list, tuple)):
    assert len(res) == 1
    return res[0], padding
  else:
    return tuple(res), padding


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
              tf.math.logical_or(
                  tf.equal(padding, 0.0), tf.equal(padding, 1.0))), [padding])
  ], padding)
  if use_select:
    if padded is None:
      padded = tf.zeros_like(x)
    if broadcast:
      # Broadcast padding to the full shape.
      padding = tf.cast(padding, x.dtype) * tf.ones_like(x)
    return tf.where(padding > tf.zeros_like(padding), padded, x)
  else:
    result = x * tf.cast(1.0 - padding, x.dtype)
    if padded is not None:
      result += padded * tf.cast(padding, padded.dtype)
    return result


def LengthsFromPaddings(paddings):
  """Computes lengths of each sequence in a batch, ignoring trailing padding.

  Note the following isn't guaranteed due to leading paddings.
  PaddingsFromLengths(LengthsFromPaddings(x)) == x

  Args:
    paddings: a tensor with shape [batch, length].

  Returns:
    lengths tensor shaped [batch] containing the unpadded length of each
    sequence in the batch.
  """
  paddings = HasRank(paddings, 2)
  paddings = tf.cast(paddings, tf.int32)
  # Find the last unpadded value.
  # Cannot just use tf.reduce_sum because there might be leading paddings.
  # Everything after the last unpadded value has 1.0 - paddings == 0.0, so in
  # the cumsum below they will have the same value.
  cumsum = tf.cumsum(1 - paddings, axis=1)
  same_as_last_element = tf.equal(cumsum, cumsum[:, -1:])
  # Counting the number of elements with the same value gives us num_padded + 1
  # and so counting the number that differs gives us num_padded - 1.
  length = tf.reduce_sum(
      1 - tf.cast(same_as_last_element, tf.int32), axis=1) + 1
  # Special case for all 0 paddings.
  all_zero_paddings = tf.equal(tf.reduce_sum(1 - paddings, axis=1), 0)
  return tf.where(all_zero_paddings, tf.zeros_like(length), length)


def PaddingsFromLengths(lengths, maxlen=None):
  """Computes paddings Tensor from lengths.

  Note the following isn't guaranteed due to leading paddings.
  PaddingsFromLengths(LengthsFromPaddings(x)) == x.

  This method does not generate leading paddings.

  Args:
    lengths: A int32 Tensor of shape [B].
    maxlen: None or a Python int or a scalar Tensor.

  Returns:
    A 0/1 valued Tensor of shape [B, maxlen or ?] where 1s are padded positions.
  """
  lengths = HasRank(lengths, 1)
  if maxlen is not None:
    lengths = with_dependencies(
        [assert_less_equal(tf.cast(tf.reduce_max(lengths), tf.int32), maxlen)],
        lengths)

  return 1. - tf.sequence_mask(lengths, maxlen=maxlen, dtype=tf.float32)


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
  max_length = tf.maximum(tf.reduce_max(LengthsFromPaddings(paddings)), 1)
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
      tf.math.rint(tf.reduce_sum(inversed_paddings, axis=0)), tf.int32)
  return tf.reverse_sequence(inputs, inputs_length, seq_axis=0, batch_axis=1)


def ConcatenatePaddedSequences(input0, input1, padding0, padding1, seq_dim=1):
  """Concatenates input sequences with varying lengths as defined by paddings.

  This is a helper function for concatenating 2 batches of input sequences,
  where each example in the batch can have different lengths, as defined by
  the corresponding paddings. To concatenate correctly, it makes use of
  tf.reverse_sequence to partially reverse the sequences before
  concatenating them together.

  NOTE: We assume that the tensors have no leading paddings.

  Args:
    input0: A tensor of size [batch, max_length, ...] or [max_length, batch,
      ...] depending on the value set for axis.
    input1:  A tensor of size [batch, max_length, ...] or [max_length, batch,
      ...] depending on the value set for axis.
    padding0: A Tensor of size [batch, max_length] or [max_length, batch]
      corresponding to the padding for input0.
    padding1: A Tensor of size [batch, max_length] or [max_length, batch]
      corresponding to the padding for input1.
    seq_dim: int, the time axis along which the tensors will be concatenated.
      Should be 0 or 1. Assumes that batch_dim is 1 - seq_dim.

  Returns:
    The concatenation of input0 and input1, and the corresponding padding.

  Raises:
    tf.errors.InvalidArgumentError when seq_dim is not 0 or 1.
  """
  if seq_dim != 0 and seq_dim != 1:
    raise tf.errors.InvalidArgumentError(None, None, 'seq_dim must be 0 or 1.')
  batch_dim = 1 - seq_dim
  # inpu0 and input1 should have the same batch size and same rank.
  input0 = with_dependencies([
      assert_equal(GetShape(input0)[batch_dim],
                   GetShape(input1)[batch_dim]),
      assert_equal(GetRank(input0), GetRank(input1))
  ], input0)

  batch_size = GetShape(padding0)[batch_dim]
  # batch dimension of inputs and paddings should match.
  input0 = with_dependencies([
      assert_equal(GetShape(input0)[batch_dim], batch_size),
      assert_equal(GetShape(padding1)[batch_dim], batch_size)
  ], input0)
  input0_seq_dim = tf.cast(
      tf.tile([tf.shape(padding0)[seq_dim]], [batch_size]), dtype=tf.int32)
  input1_seq_dim = tf.cast(
      tf.tile([tf.shape(padding1)[seq_dim]], [batch_size]), dtype=tf.int32)
  # LengthsFromPaddings assumes that paddings is of size [batch, max_length].
  if seq_dim == 1:
    seq_length0 = LengthsFromPaddings(padding0)
    seq_length1 = LengthsFromPaddings(padding1)
  else:
    seq_length0 = LengthsFromPaddings(tf.transpose(padding0))
    seq_length1 = LengthsFromPaddings(tf.transpose(padding1))
  # We assume that the tensors have no leading paddings.
  # TODO(arunnt): Concatenate tensors with leading paddings correctly.
  seq_length0 = with_dependencies([
      assert_equal(
          seq_length0,
          tf.cast(tf.reduce_sum(1.0 - padding0, seq_dim), dtype=tf.int32))
  ], seq_length0)
  seq_length1 = with_dependencies([
      assert_equal(
          seq_length1,
          tf.cast(tf.reduce_sum(1.0 - padding1, seq_dim), dtype=tf.int32))
  ], seq_length1)
  # Concatenate input sequences.
  reversed_input0 = tf.reverse_sequence(
      input0, seq_length0, seq_axis=seq_dim, batch_axis=batch_dim)
  reversed_input1 = tf.reverse_sequence(
      input1, input1_seq_dim, seq_axis=seq_dim, batch_axis=batch_dim)
  reversed_concat = tf.concat([reversed_input1, reversed_input0], axis=seq_dim)
  concat_inputs = tf.reverse_sequence(
      reversed_concat,
      seq_length0 + input1_seq_dim,
      seq_axis=seq_dim,
      batch_axis=batch_dim)
  # Concatenate paddings. Note that paddings are always a Tensor of 0s and 1s,
  # so, unlike the inputs, we don't have to reverse padding1, we can simply
  # concatenate reversed padding0 and padding1.
  reversed_padding0 = tf.reverse_sequence(
      padding0, input0_seq_dim, seq_axis=seq_dim, batch_axis=batch_dim)
  reversed_concat_padding = tf.concat([reversed_padding0, padding1],
                                      axis=seq_dim)
  concat_paddings = tf.reverse_sequence(
      reversed_concat_padding,
      input0_seq_dim + seq_length1,
      seq_axis=seq_dim,
      batch_axis=batch_dim)
  return concat_inputs, concat_paddings


def ShiftLeft(tensor, shift_size, pad_val=0, axis=1):
  """Shifts the values in a tensor to the left along the axis dimension.

  The first shift_size values are dropped, and the tensor is padded on the
  right with pad_val.

  Args:
    tensor: the input tensor with the axis dim being time.
    shift_size: the number of frames >= 0 to shift.
    pad_val: the value to pad on the right of the tensor.
    axis: The dimension along which the tensor will be shifted, default to 1.

  Returns:
    A left shifted tensor on dimension axis.
  """
  rank = tensor.shape.rank
  with tf.control_dependencies(
      [assert_greater_equal(rank, 2),
       assert_greater_equal(shift_size, 0)]):
    time = GetShape(tensor)[axis]
    begin = tf.scatter_nd([[axis]], [shift_size], [rank])
    return PadSequenceDimension(
        tf.slice(tensor, begin, size=[-1] * rank), time, pad_val, axis=axis)


def Retry(*args, **kwargs):
  return retry.Retry(*args, **kwargs)


# FailedPreconditionError: variables are not initialized.
# AbortedError: processes restarts.
# UnavailableError: Bad hardware status: 0x1
transient_tf_errors = (tf.errors.FailedPreconditionError,
                       tf.errors.AbortedError, tf.errors.UnavailableError)


def RetryOnTransientTfError(*args, **kwargs):
  return Retry(transient_tf_errors, *args, **kwargs)


def PadOrTrimTo(x, shape, pad_val=0, pad_after_contents=True):
  """Pad and slice x to the given shape.

  Args:
    x: A tensor.
    shape: The shape of the returned tensor.
    pad_val: An int or float used to pad x.
    pad_after_contents: Whether to pad and trim after the original contents of
      each dimension.

  Returns:
    'x' is padded with pad_val and sliced so that the result has the given
    shape.

  Raises:
    ValueError: if shape is a tf.TensorShape and not fully defined.
  """
  if isinstance(shape, (list, tuple)):
    expected_rank = len(shape)
  elif isinstance(shape, tf.TensorShape):
    if not shape.is_fully_defined():
      raise ValueError('shape %s padding %s must be fully defined.' %
                       (shape, x))
    expected_rank = shape.rank
  else:
    shape = HasRank(shape, 1)
    expected_rank = tf.size(shape)
  x = HasRank(x, expected_rank)

  pad = shape - tf.minimum(tf.shape(x), shape)
  zeros = tf.zeros_like(pad)
  if pad_after_contents:
    # If dim_i is less than shape[i], pads after contents.
    paddings = tf.stack([zeros, pad], axis=1)
    # If dim_i is larger than shape[i], we slice [0:shape[i]] for dim_i.
    slice_begin = zeros
  else:
    # If dim_i is less than shape[i], pads before contents.
    paddings = tf.stack([pad, zeros], axis=1)
    # If dim-i is larger than shape[i], we slice [dim_i - shape[i]:dim_i]
    # for dim_i.
    slice_begin = tf.shape(x) + pad - shape

  x = tf.pad(x, paddings, constant_values=pad_val)
  x = tf.slice(x, slice_begin, shape)

  return tf.reshape(x, shape)


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


def MixByWeight(inputs, weights, seed=None):
  """Returns a weighted random choice and bprop type from the give inputs.

  Args:
    inputs: a list of callables, where each callable returns a tf.Tensor or a
      nested structure containing tf.Tensor. Function return types must be
      consistent across elements. The tf.Operation to compute the result tensor
      will only be invoked for one input at a time. For example, if each fn
      represents an input record stream, a record will be drawn only from a
      selected stream while the other streams will remain unchanged.
    weights: a 1D tensor of float > 0 of the same length as inputs.
    seed: random seed.

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
  r = tf.random.uniform(shape=[], maxval=upper[-1], seed=seed)
  return_input = tf.case(
      [(tf.math.logical_and(lower[i] <= r, r < upper[i]), inputs[i])
       for i in range(len(inputs))],
      exclusive=True)
  selected_index = tf.case(
      [(tf.math.logical_and(lower[i] <= r, r < upper[i]), lambda i=i: i)
       for i in range(len(inputs))],
      exclusive=True)
  bprop_index = tf.one_hot(selected_index, len(inputs), dtype=tf.float32)
  return return_input, bprop_index


def CheckShapes(shapes):
  """Asserts that shapes is a tuple of NestedMap or tshape.Shape."""
  assert isinstance(shapes, tuple), str(shapes)
  for s in shapes:
    if isinstance(s, NestedMap):
      assert all([isinstance(t, tshape.Shape) for t in Flatten(s)
                 ]), '{} contains non-tensor value.'.format(s)
    else:
      assert isinstance(s, tshape.Shape), '{}: {}'.format(type(s), s)


def FPropDtype(params):
  return params.fprop_dtype if params.fprop_dtype is not None else params.dtype


def UpdateFpropDtype(params, fprop_dtype):
  """Recursively update the fprop_dtype of the Params."""
  # Handle the case when the input "params" is not an instance of hyperparams
  # For example, when UpdateDtype is called recursively for all the items in
  # the "sub" list of SequentialLayer (see 1st elif below)
  if not isinstance(params, hyperparams.Params):
    return

  for key, val in params.IterParams():
    if isinstance(val, hyperparams.Params):
      UpdateFpropDtype(val, fprop_dtype)
    elif isinstance(val, (list, tuple)):
      for item in val:
        UpdateFpropDtype(item, fprop_dtype)
    elif key == 'fprop_dtype':
      params.fprop_dtype = fprop_dtype


def UpdateDtype(params, dtype):
  """Recursively update the dtype of the Params."""
  # Handle the case when the input "params" is not an instance of hyperparams
  # For example, when UpdateDtype is called recursively for all the items in
  # the "sub" list of SequentialLayer (see 1st elif below)
  if not isinstance(params, hyperparams.Params):
    return

  for key, val in params.IterParams():
    if isinstance(val, hyperparams.Params):
      UpdateDtype(val, dtype)
    elif isinstance(val, (list, tuple)):
      for item in val:
        UpdateDtype(item, dtype)
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


# TODO(jamesqin): follow suggestions in
# b/167460492#comment16
def RematerializeFn(fn, *xs):
  """Calls fn and rematerializes fn in the backward pass.

  `fn(*xs) -> ys`, where xs and ys can be a single tensor or a tuple of tensors.

  Args:
    fn: A python function to be rematerialized in the backprop pass.
    *xs: A single tensor or a list/tuple of tensors. `xs` are input args to the
      fn function.

  Returns:
    `fn(*xs)`
  """
  initial_step_seed = GetStepSeed()
  final_step_seed = MaybeGenerateSeedFromScope()

  def Backward(fwd_xs, fwd_ys, d_fwd_ys):
    """The backward function that rematerializes forward outputs."""
    del fwd_ys
    always_true = tf.random.uniform([]) < 2.0
    # Alternatively, can do this:
    # tf.where(tf.math.is_nan(x),
    #          tf.constant(float('nan'), dtype=x.dtype) * tf.ones_like(x),
    #          x)
    bak_xs = [tf.where(always_true, x, tf.zeros_like(x)) for x in fwd_xs.xs]
    for dst, src in zip(bak_xs, xs):
      dst.set_shape(src.shape)
    ResetStepSeed(initial_step_seed)
    ys = fn(*bak_xs)
    MaybeResetStepSeed(final_step_seed)
    dxs = tf.gradients(ys, bak_xs, grad_ys=d_fwd_ys)
    dxs_final = []
    for dx, x in zip(dxs, bak_xs):
      if dx is None:
        dxs_final.append(tf.zeros_like(x))
      else:
        dxs_final.append(dx)
    assert len(dxs_final) == len(bak_xs)
    return NestedMap(
        initial_step_seed=tf.zeros_like(initial_step_seed), xs=dxs_final)

  ys_shapes = []

  # TODO(huangyp, yonghui): Check Forward doesn't use any stateful random ops.
  def Forward(fwd_xs):
    """Forward function plus sanity checks."""
    for dst, src in zip(fwd_xs.xs, xs):
      dst.set_shape(src.shape)
    ResetStepSeed(fwd_xs.initial_step_seed)
    ys = fn(*fwd_xs.xs)
    # Some sanity check.
    assert not GetExtraInputs()
    assert not GetExtraArgs()
    assert not GetExtraVars()
    if isinstance(ys, tuple):
      for y in ys:
        assert isinstance(y, tf.Tensor)
        ys_shapes.append(y.shape)
    else:
      assert isinstance(ys, tf.Tensor)
      ys_shapes.append(ys.shape)
    return ys

  ys = CallDefun(
      Forward,
      NestedMap(initial_step_seed=initial_step_seed, xs=xs),
      bak=Backward)
  if isinstance(ys, tuple):
    for y, s in zip(ys, ys_shapes):
      y.set_shape(s)
  else:
    ys.set_shape(ys_shapes[0])
  # TODO(b/129159299): The ResetStepSeed below is needed to work around this
  # bug, which is a problem with global tensors being shared by different
  # inference graphs. It should be replaced with the new step seed value
  # returned from the Forward function when the bug is fixed.
  MaybeResetStepSeed(final_step_seed)
  return ys


# A set of names of stateful random number generator ops.
# See tensorflow/core/ops/random_ops.cc
_STATEFUL_RANDOM_OPS = frozenset({
    # pyformat: disable
    'RandomUniform',
    'RandomUniformInt',
    'RandomStandardNormal',
    'ParameterizedTruncatedNormal',
    'TruncatedNormal',
    'RandomShuffle',
    'Multinomial',
    'RandomGamma',
    'RandomPoisson',
    'RandomPoissonV2',
    # pyformat: enable
})


def StatefulRandomOpsInDefun(func, graph=None):
  """Checks whether the Defun depends on stateful random number ops.

  Stateful random number generator ops should be avoid in Recurrent() call.
  Otherwise, these ops produce inconsistent values between FProp and BProp.

  Args:
    func: a _DefinedFunction or ConcreteFunction to check.
    graph: a Graph. Set None to use the default graph.

  Returns:
    A list of names of the stateful random ops.

  Raises:
    InvalidArgumentError: if the input func/graph is invalid.
  """
  if graph is None:
    graph = tf.get_default_graph()
  func.add_to_graph(graph)
  graph_def = graph.as_graph_def()

  # A dict from function name to FunctionDef.
  func_defs = {x.signature.name: x for x in graph_def.library.function}

  if isinstance(func, function._DefinedFunction):  # pylint: disable=protected-access
    if func.definition.signature.name not in func_defs:
      raise tf.errors.InvalidArgumentError(
          None, None, 'Defun {} is not in the graph .'.format(
              func.definition.signature.name))
    nodes = py_collections.deque(func.definition.node_def)
  else:
    nodes = py_collections.deque(func.function_def.node_def)

  stateful_ops = []

  # Recursively search for stateful random op.
  while nodes:
    node = nodes.pop()
    assert isinstance(node, node_def_pb2.NodeDef), node

    if node.op in _STATEFUL_RANDOM_OPS:
      stateful_ops.append(node.name)
      continue

    def _AddDefunNodes(func_name):
      """If the given func_name is a Defun, add its sub-nodes into nodes."""
      if func_name in func_defs:
        nodes.extend(func_defs[func_name].node_def)

    # For functional.{While|For|If} ops, add their Defun attr into search.
    if node.op == 'While':
      _AddDefunNodes(node.attr['body'].func.name)
      _AddDefunNodes(node.attr['cond'].func.name)
    elif node.op == 'For':
      _AddDefunNodes(node.attr['body'].func.name)
    elif node.op == 'If':
      _AddDefunNodes(node.attr['then_branch'].func.name)
      _AddDefunNodes(node.attr['else_branch'].func.name)
    elif node.op == 'StatefulPartitionedCall':
      _AddDefunNodes(node.attr['f'].func.name)
    elif node.op != 'PartitionedCall':
      # For other op, check whether itself is a Defun op.
      _AddDefunNodes(node.op)

  return stateful_ops


def ToPlaceholders(nmap, dtype=None):
  """Converts every Tensor in nmap to a placeholder."""

  def _ToPlacerholder(x):
    shape = [None for _ in x.shape[:-1]] + [x.shape[-1]]
    return tf.placeholder(dtype=dtype or x.dtype, shape=shape)

  return nmap.Transform(_ToPlacerholder)


def Softmax(logits, axis=None, extra_logit=None, name=None):
  """Softmax with extra_logits, might be useful for large xformer LM."""
  if extra_logit is None:
    return tf.nn.softmax(logits, axis=axis, name=name)

  axis = -1 if axis is None else axis

  def ReduceLogSumExp(x):
    max_logit = tf.math.reduce_max(
        tf.stop_gradient(x), axis=axis, keepdims=True)

    base_logit = tf.math.maximum(max_logit, extra_logit)
    x -= base_logit
    exp_x = tf.math.exp(x)
    sum_exp_x = tf.math.reduce_sum(exp_x, axis=axis, keepdims=True)

    sum_exp_x += tf.math.exp(extra_logit - base_logit)
    return tf.math.log(sum_exp_x) + base_logit

  def LogSoftmax(x):
    return x - ReduceLogSumExp(x)

  with tf.name_scope(name):
    return tf.math.exp(LogSoftmax(logits))


def SoftmaxCrossEntropyFocalLoss(logits,
                                 label_ids=None,
                                 label_probs=None,
                                 alpha=None,
                                 gamma=None,
                                 stop_gradient_on_focal_loss_coefficient=False):
  u"""Focal loss for multinomial (softmax) logistic loss.

  [1] Focal loss https://arxiv.org/abs/1708.02002

  Args:
    logits: [..., C]. Logits for the multinomial logistic regression. C is the
      number of classes.
    label_ids: [...]. Each entry in labels must be an index in [0, C).
    label_probs: [..., C]. Each vector along last dimension must be a valid
      probability distribution.
    alpha: [C]. The weighting factor alpha. Eq (3) in [1].
    gamma: []. Tunable focusing parameter. Eq (4) in [1].
    stop_gradient_on_focal_loss_coefficient: If true, stops gradient on the
      focal loss coefficient (1-p)^gamma to stabilize the gradient.

  Returns:
    loss[i..., j] = FL(pₜ) = - αₜ(1-pₜ)ˠlog(pₜ) Eq (5) in [1].
  """

  def _ApplyFocalLossCoefficient(loss, log_probs):
    if gamma is not None and gamma != 0:
      probs = tf.exp(log_probs)
      coefficient = tf.pow(1.0 - probs, gamma)
      if stop_gradient_on_focal_loss_coefficient:
        coefficient = tf.stop_gradient(coefficient)
      loss *= coefficient
    return loss

  if label_probs is not None:
    log_probs = tf.nn.log_softmax(logits)
    loss = -(label_probs * log_probs)
    loss = _ApplyFocalLossCoefficient(loss, log_probs)
    if alpha is not None:
      loss *= tf.reshape(
          alpha, tf.concat([tf.ones(tf.rank(loss) - 1, tf.int32), [-1]],
                           axis=0))
    loss = tf.reduce_sum(loss, axis=-1)
  else:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_ids, logits=logits)
    loss = _ApplyFocalLossCoefficient(loss, -loss)
    if alpha is not None:
      loss *= tf.gather(alpha, label_ids)
  return loss


def SigmoidCrossEntropyFocalLoss(logits, labels, alpha=None, gamma=None):
  u"""Focal loss for binary (sigmoid) logistic loss.

  [1] Focal loss https://arxiv.org/abs/1708.02002

  Args:
    logits: [..., C]. Logits for the sigmoid logistic regression.
    labels: [..., C]. 0/1 labels.
    alpha: The weighting factor alpha. Eq (3) in [1].
    gamma: Tunable focusing parameter. Eq (4) in [1].

  Returns:
    loss[i..., j] = FL(pₜ) = - αₜ(1-pₜ)ˠlog(pₜ) Eq (5) in [1].
  """

  # [1] Eq (4).
  #
  # The numerically-stable way to compute
  #  log(p) for positives;
  #  log(1 - p) for negatives.
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  if gamma is not None and gamma != 0:
    # The modulating factor. Note that
    # (1 - p)ˠ = [1 - σ(x)]ˠ = [σ(-x)]ˠ, for positives.
    # pˠ = [σ(x)]ˠ, for negatives.
    loss *= tf.pow(tf.sigmoid(logits * (1 - labels * 2)), gamma)

  if alpha is not None:
    # [1] Eq (3)
    loss *= (alpha * labels + (1 - alpha) * (1 - labels))

  return loss


_RECORD_FORMAT_RE = re.compile('(^[A-Za-z]+):(.*)')


def RecordFormatFromFilePattern(file_pattern):
  """Return the record format string for a Lingvo file pattern.

  Lingvo file patterns take the form of:
    tfrecord:/path/to/bar -> tfrecord is the record_format.

  This function takes a file pattern and returns a string indicating
  which format the filepattern implies.

  Args:
    file_pattern: String file pattern.

  Returns:
    Tuple (string, string):

      - record_format: String record format, e.g., "tfrecord", etc.
      - file_pattern: The file pattern without any prefixes.
  """
  result = re.match(_RECORD_FORMAT_RE, file_pattern)

  if result is None:
    # TODO(vrv): Fix all callers so that file_pattern must contain
    # the record format prefix.
    return 'sstable', file_pattern

  # regexp ensures that a match implies there are two groups:
  # the record format and then the file pattern.
  return result.groups()


def ReadFileLines(file_path):
  """Read a text file and return the lines.

  If the file cannot be found at the given path, attempt to load it from the
  Lingvo package (useful for data dependencies in par files).

  Args:
    file_path: path to file, either absolute or relative to the bazel workspace.

  Returns:
    A list of lines from the file.
  """
  if not tf.io.gfile.exists(file_path):
    try:
      lines = pkgutil.get_data(
          'lingvo', file_path.replace('lingvo/', '',
                                      1)).splitlines(True)
    except IOError:
      # If pkgutil can't find the file, continue and let GFile raise the error.
      lines = None
  else:
    lines = None

  if not lines:
    with tf.io.gfile.GFile(file_path, 'r') as f:
      lines = f.readlines()

  return lines


# Partially borrowed from
# https://github.com/tensorflow/tensor2tensor/blob/32929305e1a4ec926eff24123758b794df35492b/tensor2tensor/layers/common_layers.py#L349
def CumSum(x, axis=0, exclusive=False):
  """A TPU efficient implementation of tf.cumsum().

  This is equivalent to tf.cumsum and is faster on TPU as of 08/2019 unless
  the axis dimension is very large. The current Tensorflow implementation is
  based on scanning and reducing which is not efficient on TPU.

  Args:
    x: An input Tensor.
    axis: An int for the axis.
    exclusive: A bool for performing exclusive cumsum.

  Returns:
    A Tensor of the same shape as x.

  Raises:
    ValueError: if the input axis is invalid.
  """
  if x.dtype not in (tf.float32, tf.bfloat16) or not use_tpu():
    # Fallback to tf.cumsum when inputs are not floats or not running on TPU.
    return tf.cumsum(x, axis=axis, exclusive=exclusive)

  rank = GetRank(x)
  # Needs to know the rank for the final transpose if axis is not the last
  # dimension. Otherwise, falls back to tf.cumsum.
  if not isinstance(rank, int) and axis != -1:
    return tf.cumsum(x, axis=axis, exclusive=exclusive)

  if axis < -1:
    if axis + rank < 0:
      raise ValueError('Unexpected axis: %d (rank = %d)' % (axis, rank))
    axis += rank

  length = GetShape(x)[axis]
  my_range = tf.range(length)
  comparator = tf.less if exclusive else tf.less_equal
  mask = tf.cast(
      comparator(tf.expand_dims(my_range, 1), tf.expand_dims(my_range, 0)),
      x.dtype)
  result = tf.tensordot(x, mask, axes=[[axis], [0]])
  if axis != -1 and axis != rank - 1:
    result = tf.transpose(
        result,
        list(range(axis)) + [rank - 1] + list(range(axis, rank - 1)))
  return result


def ProjectLastDim(inputs, weight, input_dim, output_dim):
  """Linear projection on the last dim of the input tensor.

  This is a TPU efficient implementation to avoid reshaping inputs to Rank-2
  tensor by using Einsum for the compute.

  Args:
    inputs: An input Tensor, the last dimension of which is input_dim.
    weight: A weight matrix with shape [input_dim, output_dim].
    input_dim: An integer or a symbolic dim, the last dimension of the inputs.
    output_dim: An integer or a symbolic dim, the last dimension of the outputs.

  Returns:
    An output Tensor of the same rank as inputs, the last dimension is
    output_dim.
  """
  input_dim = int(
      symbolic.ToStatic(input_dim) if symbolic.IsExpr(input_dim) else input_dim)
  output_dim = int(
      symbolic.ToStatic(output_dim) if symbolic.IsExpr(output_dim
                                                      ) else output_dim)

  # Assert input_dim and output_dim
  inputs = with_dependencies([assert_equal(GetShape(inputs)[-1], input_dim)],
                             inputs)
  weight = with_dependencies([
      assert_equal(GetShape(weight)[0], input_dim),
      assert_equal(GetShape(weight)[-1], output_dim)
  ], weight)

  if (use_tpu() and inputs.shape is not None and
      inputs.shape.rank is not None and inputs.shape.rank < 26):
    # Avoids reshape if feasible and uses Einsum.
    if inputs.shape.rank == 2:
      outputs = tf.matmul(inputs, weight)
    else:
      # This is equivalent to:
      #   outputs = tf.einsum('...y,yz->...z', inputs, weight)
      # Unfortunately ... in einsum() leads to extra HBM usage.
      s = ''.join([chr(x) for x in range(97, 123)])  # abc...xyz
      r = inputs.shape.rank
      outputs = tf.einsum('{0}y,yz->{0}z'.format(s[:r - 1]), inputs, weight)
  else:
    outputs = Matmul(tf.reshape(inputs, ToStaticShape([-1, input_dim])), weight)
    outputs = tf.reshape(
        outputs,
        tf.concat([
            tf.cast(GetShape(inputs)[:-1], tf.int32),
            ToStaticShape([output_dim])
        ],
                  axis=0))

  return outputs


@contextlib.contextmanager
def RemoveAssertContext(remove=True):
  """Hacks to replace certain unwanted tensorflow ops."""
  # TODO(zhifengc/huangyp): Consider implementing assert_equal
  # op replacement for lingvo. As assert_equal doesn't support String on GPUs.
  # Hack to replace tf.assert_equal
  # TODO(b/136040013): Remove this after migration to tf.function.
  if remove:
    saved_assert_equal = tf.check_ops.assert_equal

    def NoOP(*args, **kwargs):  # pylint: disable=unused-argument
      return tf.no_op()

    tf.check_ops.assert_equal = NoOP  # Make assert_equal a no op.
    try:
      yield
    finally:
      tf.check_ops.assert_equal = saved_assert_equal
  else:
    yield


def _AssertInputsMatch(op, args, implicit_captures):
  """Assert that op's inputs match with args and implicit_captures.

  Args:
    op: The operation to check.
    args: A nested structure representing the explicit arguments of 'op'.
    implicit_captures: A nested structure representing the implicitly captured
      inputs of 'op'.

  Raises:
    ValueError: if the number of inputs mismatch.
  """
  expected_inputs = Flatten([args, implicit_captures])
  expected_num_inputs = len(expected_inputs)
  if len(op.inputs) > expected_num_inputs:
    raise ValueError(('Too many inputs. The most likely cause is that fwd '
                      'captures additional tensors: extra inputs %r vs %r '
                      'captures=%r') % (list(op.inputs), list(expected_inputs),
                                        list(Flatten(implicit_captures))))
  if len(op.inputs) < expected_num_inputs:
    raise ValueError(('Mismatched inputs to fwd: Found %d vs expected %d: %r'
                      '. Implicit captures(%d) = %r') %
                     (len(op.inputs), expected_num_inputs, list(op.inputs),
                      len(Flatten(implicit_captures)), implicit_captures))


def TensorSpecs(nmap, keep_shape=True):
  """Transforms tensors in the input nested structure to TensorSpecs."""
  if nmap is None:
    return None
  fn = lambda t: tf.TensorSpec(t.shape if keep_shape else None, t.dtype)
  return Transform(fn, nmap)


def _DefineDefun(fwd, fwd_sig, bak=None, bak_as_function=False, device=None):
  """Wraps fwd in a defun with custom gradient bak.

  Args:
    fwd: A callable xs: Nested Structure -> ys: Nested Structure.
    fwd_sig: A Nested Structure of tf.TensorSpec representing the input
      signature of `fwd`, or None (meaning that fwd takes no inputs).
    bak: A callable xs, ys, dys: Nested Structure -> dxs[, dcapture]: Nested
      Structure. The custom backprop function for `fwd`. bak needs to return
      dcapture if fwd uses any implicitly captured tensors, whose gradients are
      dcapture.
    bak_as_function: Whether to create a TF graph function for `bak`.
    device: the device on which to run `fwd` and `bak`.

  Returns:
    A NestedMap containing:

    - call: A callable that will execute `fwd`. It has the same input and output
      signatures as `fwd`.
    - func: The underlying TF function that `call` calls. If not None, it will
      be a _DefinedFunction or ConcreteFunction that takes flat inputs and
      returns flat outputs, and can be used by routines that require a TF
      function object (e.g. tf.If, tf.While, etc).
      Always not None when `bak` is None.
    - output_dtypes: A nested structure compatible with the outputs of `fwd`
      containing the corresponding output dtypes.
    - stateful_ops: A list of (op_name, op_type) tuples representing the
      stateful ops used by `fwd`.
    - captured_inputs: Implicit inputs captured by `fwd`.
  """
  assert fwd is not None
  noinline = not use_xla()

  if fwd_sig is None:
    fwd_sig = []
  get_dtype = lambda x: x.dtype
  arg_dtypes = Flatten(Transform(get_dtype, fwd_sig))
  get_shape = lambda x: x.shape
  arg_shapes = Flatten(Transform(get_shape, fwd_sig))

  # Used to hold the backward function used by Grad, which will be defined if
  # bak is set.
  sigs = NestedMap()
  # Output of this method.
  res = NestedMap()

  python_grad_func = None
  if bak:

    def Grad(op, *args):
      """Gradient function for the forward function.

      Args:
        op: The forward operation.
        *args: Gradients wrt op.outputs.

      Returns:
        Tuple of derivatives.
      """
      _AssertInputsMatch(op, fwd_sig, res.captured_inputs)
      # Ensure dys contains no None.
      args = ConvertNoneGradientToZeros(list(op.outputs), list(args))
      xs = op.inputs[:len(arg_dtypes)]  # The rest are captures.
      return sigs.backward(*Flatten([xs, op.outputs, args]))

    python_grad_func = Grad

  def _SetShape(dst_list, shape_list):
    for dst, shape in zip(dst_list, shape_list):
      if isinstance(dst, tf.Tensor):
        dst.set_shape(shape)

  @tf.Defun(*arg_dtypes, python_grad_func=python_grad_func, noinline=noinline)
  def Forward(*args):
    """The forward function."""
    _SetShape(args, arg_shapes)
    with RemoveAssertContext(remove=noinline):
      call = lambda: fwd(Pack(fwd_sig, args)) if args else fwd()
      if device is None:
        # Defun will handle the device assignment.
        rets = call()
      else:
        with tf.device(device):
          rets = call()
    res.outputs = rets
    return Flatten(rets)

  forward = Forward
  if not arg_dtypes:
    # In this case Forward is an _OverloadedFunction, we need to instantiate it.
    forward = Forward.instantiate([])

  # Invokes fwd() to get res.outputs.
  forward.add_to_graph(tf.get_default_graph())
  res.func = forward
  res.stateful_ops = forward.stateful_ops
  res.captured_inputs = forward.captured_inputs
  output_dtypes = Transform(get_dtype, res.outputs)
  output_shapes = Transform(get_shape, res.outputs)

  def Call(args=None):
    """Wrapper of fwd."""
    if args is None:
      flat_rets = forward()
    else:
      flat_rets = forward(*Flatten(args))
    if not isinstance(flat_rets, (tuple, list)):
      flat_rets = [flat_rets]
    _SetShape(flat_rets, Flatten(output_shapes))
    return Pack(output_dtypes, flat_rets)

  res.call = Call

  if bak:

    def Backward(*args):
      """The backward function."""
      _SetShape(args, Flatten([arg_shapes, output_shapes, output_shapes]))
      xs, ys, dys = Pack([fwd_sig, output_dtypes, output_dtypes], args)
      with RemoveAssertContext(remove=noinline):
        if device is None:
          # Defun will handle the device assignment.
          dxs = bak(xs, ys, dys)
        else:
          with tf.device(device):
            dxs = bak(xs, ys, dys)
      return Flatten(dxs)

    if bak_as_function:
      sigs.backward = tf.Defun(
          *Flatten([arg_dtypes, output_dtypes, output_dtypes]),
          noinline=noinline)(
              Backward)

      sigs.backward.add_to_graph(tf.get_default_graph())
    else:
      sigs.backward = Backward

  return res


# Global variable to control rendezvous sharing in tf.function.
# If False (default) rendezvous sharing is disabled in tf.function, that is, the
# function body use a separate rendezvous and can't communicate with parent
# graph via send/recv.
# With _GetSharedRendezvous() == True, the function body share the same
# rendezvous with the parent graph and can talk to it using send/recv. This is
# useful for layers like StackedRecurrent.
_SHARED_RENDEZVOUS = ThreadLocalStack()


@contextlib.contextmanager
def _SharedRendezvousScope(shared_rendezvous=True):
  _SHARED_RENDEZVOUS.stack.append(shared_rendezvous)
  try:
    yield
  finally:
    _SHARED_RENDEZVOUS.stack.pop()


def _GetSharedRendezvous():
  """Get the current rendezvous sharing setting."""
  return _SHARED_RENDEZVOUS.stack[-1] if _SHARED_RENDEZVOUS.stack else False


def _ApplySharedRendezvous(func):
  """Apply the rendezvous sharing setting on the given tf.function func."""
  # pylint: disable=protected-access
  func._shared_rendezvous = _GetSharedRendezvous()
  # pylint: enable=protected-access


def _WrapFunction(func=None, input_signature=None):
  """Wraps func as a tf.function."""
  if input_signature is None:
    input_signature = []

  def Decorated(fn):

    @tf.function(input_signature=input_signature, autograph=False)
    def Fn(*args):
      # TODO(b/163904067): mimic Defun' behavior and reset the step seed to
      # avoid it being used as an implicit capture. This is not a desired
      # behavior, it should take the step seed from parent graph instead.
      ResetStepSeed()

      # Mimic Defun and disable collection sharing.
      graph = tf.get_default_graph()
      # Don't share summaries collection with parent graph (b/168745134).
      graph.clear_collection(tf.GraphKeys.SUMMARIES)
      return fn(*args)

    _ApplySharedRendezvous(Fn)

    # Add the function to the graph so it'll be traced under the current
    # context. This is necessary if the function body captures any non-tensor
    # values from the environment, like symbolic maps.
    cf = Fn.get_concrete_function()
    cf.add_to_graph()
    return cf

  # For the `foo = _WrapFunction(foo, ...)` use case.
  if func is not None:
    return Decorated(func)

  # For the `@_WrapFunction(...)` use case.
  return Decorated


def _DefineFunction(fwd, fwd_sig, bak=None, bak_as_function=False, device=None):
  """Wraps fwd in a defun with custom gradient bak.

  Args:
    fwd: A callable xs: Nested Structure -> ys: Nested Structure.
    fwd_sig: A Nested Structure of tf.TensorSpec representing the input
      signature of `fwd`, or None (meaning that fwd takes no inputs).
    bak: A callable xs, ys, dys: Nested Structure -> dxs[, dcapture]: Nested
      Structure. The custom backprop function for `fwd`. bak needs to return
      dcapture if fwd uses any implicitly captured tensors, whose gradients are
      dcapture.
    bak_as_function: Whether to create a TF graph function for `bak`.
    device: the device on which to run `fwd` and `bak`.

  Returns:
    A NestedMap containing:

    - call: A callable that will execute `fwd`. It has the same input and output
      signatures as `fwd`.
    - func: The underlying TF function that `call` calls. If not None, it will
      be a _DefinedFunction or ConcreteFunction that takes flat inputs and
      returns flat outputs, and can be used by routines that require a TF
      function object (e.g. tf.If, tf.While, etc).
      Always not None when `bak` is None.
    - outputs: The outputs of `fwd`. Used for reflection only (e.g. to get the
      output dtypes, shapes, etc).
    - stateful_ops: A list of (op_name, op_type) tuples representing the
      stateful ops used by `fwd`.
    - captured_inputs: Implicit inputs captured by `fwd`.
  """
  assert fwd is not None
  noinline = not use_xla()

  if fwd_sig is None:
    fwd_sig = []

  if device is None:
    # Get the current device to mimic Defun's behavior.
    # pylint: disable=protected-access
    device_funcs = tf.get_default_graph()._device_functions_outer_to_inner
    device = device_funcs[-1] if device_funcs else None
    # pylint: enable=protected-access

  # Output of this method.
  res = NestedMap()

  @_WrapFunction(input_signature=Flatten(fwd_sig))
  def Forward(*args):
    """The forward function."""
    with RemoveAssertContext(remove=noinline), tf.device(device):
      if args:
        xs = Pack(fwd_sig, args)
        rets = fwd(xs)
      else:
        rets = fwd()
    res.outputs = rets
    return Flatten(rets)

  res.captured_inputs = Forward.captured_inputs

  # Get the stateful ops used in cell_fn. Logic borrowed from
  # _EagerDefinedFunction.__init__().
  graph = Forward.graph
  input_ops = set(arg.op for arg in graph.inputs)
  operations = [op for op in graph.get_operations() if op not in input_ops]
  res.stateful_ops = [(o.name, o.type) for o in operations if o._is_stateful]  # pylint: disable=protected-access

  def Call(func, args=None):
    """Wrapper of fwd."""
    if args is None:
      flat_rets = func()
    else:
      flat_rets = func(*Flatten(args))
    if not isinstance(flat_rets, (tuple, list)):
      flat_rets = [flat_rets]
    return Pack(res.outputs, flat_rets)

  if not bak:
    res.func = Forward
    res.call = lambda args=None: Call(Forward, args)
    return res

  shared_rendezvous = _GetSharedRendezvous()
  ret_specs = TensorSpecs(res.outputs)

  def Backward(*args):
    xs, ys, dys = Pack([fwd_sig, ret_specs, ret_specs], args)
    with RemoveAssertContext(remove=noinline), tf.device(device):
      dxs = bak(xs, ys, dys)
    return Flatten(dxs)

  if bak_as_function:
    backward_cf = _WrapFunction(
        Backward, input_signature=Flatten([fwd_sig, ret_specs, ret_specs]))
  else:

    def BackwardWithSharedRendezvous(*args):
      with _SharedRendezvousScope(shared_rendezvous):
        return Backward(*args)

    backward_cf = BackwardWithSharedRendezvous

  @tf.custom_gradient
  def ForwardWithGrad(*args):
    """Forward function and its custom gradient."""
    # Note that `args` includes implicit captures. This is required by
    # tf.custom_gradient so that when the Grad() outputs include gradients to
    # implicit captures, they match the inputs to ForwardWithGrad().
    #
    # However, Forward doesn't take implicit captures as input, so we exclude
    # them here.
    fwd_args = args[:(len(args) - len(Flatten(res.captured_inputs)))]
    op = NestedMap(inputs=args, outputs=Forward(*fwd_args))

    def Grad(*args, **kwargs):
      """Gradient function for the forward function.

      Args:
        *args: Gradients wrt op.outputs.
        **kwargs: Additional arguments from tf.custom_gradient.

      Returns:
        Tuple of derivatives.
      """
      if kwargs:
        tf.logging.warning(
            'Ignoring additional arguments used by tf.custom_gradient: %s',
            str(kwargs))

      _AssertInputsMatch(op, fwd_sig, res.captured_inputs)

      # Ensure dys contains no None.
      args = ConvertNoneGradientToZeros(list(op.outputs), list(args))

      xs, _ = Pack([fwd_sig, res.captured_inputs], op.inputs)
      return backward_cf(*Flatten([xs, op.outputs, args]))

    return op.outputs, Grad

  res.func = None
  forward = lambda *xs: ForwardWithGrad(*Flatten([xs, res.captured_inputs]))
  res.call = lambda args=None: Call(forward, args)
  return res


# Global variable to control whether to use tf.function.
# If not set, the result is determined by tf2 status. See _UseTfFunction for
# details.
# TODO(laigd): remove after b/169869929 is fixed.
_USE_TF_FUNCTION = ThreadLocalStack()

# Constants for propagating framework tensors through Function.
_FRAMEWORK_TENSOR_GLOBAL_STEP = '_global_step'


@contextlib.contextmanager
def TfFunctionScope(use_tf_function=True):
  _USE_TF_FUNCTION.stack.append(use_tf_function)
  try:
    yield
  finally:
    _USE_TF_FUNCTION.stack.pop()


def _UseTfFunction():
  """Whether to use tf.function instead of tf.Defun."""
  if _USE_TF_FUNCTION.stack:
    return _USE_TF_FUNCTION.stack[-1]
  return tf2_enabled()


class Function(object):
  """Function builds a TensorFlow graph function from a callable.

  In the high level this is similar to tf.Defun and tf.function. In fact this
  relies on those as underlying implementations, but with specific configuration
  so it's easier to use and can work well in some extreme cases in Lingvo.

  Example usage:

  - No inputs:

    >>> @Function()
    ... def foo():
    ...   return tf.constant(1.0)
    >>> y = foo()

  - Scalar input:

    >>> @Function(fwd_sig=tf.TensorSpec(None, tf.float32))
    ... def foo(x):
    ...   return x * 2
    >>> y = foo(1.0)

  - List input:

    >>> @Function(fwd_sig=[tf.TensorSpec(None, tf.float32) for _ in range(2)])
    ... def foo(xs):
    ...   return xs[0] + xs[1]
    >>> y = foo([1.0, 2.0])

  - Nested input:

    >>> @Function(fwd_sig=NestedMap(x=tf.TensorSpec(None, tf.float32)))
    ... def foo(nmap):
    ...   return nmap.x * 2
    >>> y = foo(NestedMap(x=1.0))

  - With custom gradient function (other input types mentioned above are also
    supported):

    >>> def bar(x, y, dy):
    ...   del y, dy
    ...   return 4.0 * x * dy
    >>>
    >>> @Function(fwd_sig=tf.TensorSpec(None, tf.float32), bak=bar)
    ... def foo(x):
    ...   return 2.0 * x * x

  - Used in control flow ops:

    >>> then_branch = Function(tf.TensorSpec([], tf.int32))(lambda x: x / 2)
    >>> else_branch = Function(tf.TensorSpec([], tf.int32))(lambda x: 3 * x + 1)
    >>> y = tf.If(cond, inputs, then_branch.func, else_branch.func)
  """

  # TODO(laigd): the use_tf_function option is added for backward compatibility
  # reasons. Remove it after the migration.
  def __init__(self,
               fwd_sig=None,
               bak=None,
               bak_as_function=False,
               device=None,
               use_tf_function=None):
    """Constructor.

    Below we assume `fwd` is the input to `__call__` that is used to build the
    TensorFlow graph function encapsulated by this object.

    Args:
      fwd_sig: A Nested Structure of tf.TensorSpec representing the input
        signature of `fwd`, or None (meaning that `fwd` takes no inputs). The
        actual inputs should be compatible with this (have same shapes and
        dtypes).
      bak: A callable xs, ys, dys: Nested Structure -> dxs[, dcapture]: Nested
        Structure. The custom backprop function for `fwd`. bak needs to return
        dcapture if `fwd` uses any implicitly captured tensors, whose gradients
        are dcapture.
      bak_as_function: Whether to create a TF graph function for `bak`.
      device: The device on which to run `fwd` and `bak`. Defaults to the
        current device.
      use_tf_function: Whether use tf.function. Defaults to _UseTfFunction().
    """
    self._fwd_sig = fwd_sig
    self._bak = bak
    self._bak_as_function = bak_as_function
    self._device = device
    self._use_tf_function = use_tf_function

  def __call__(self, fwd):
    """Creates a graph function.

    Args:
      fwd: a callable xs: Nested Structure -> ys: Nested Structure.

    Returns:
      A DefinedFunction object encapsulating `fwd` as a graph function.
    """
    assert callable(fwd)
    return DefinedFunction(fwd, self._fwd_sig, self._bak, self._bak_as_function,
                           self._device, self._use_tf_function)


class DefinedFunction(object):
  """Encapsulates a TensorFlow graph function and its properties."""

  def __init__(self,
               fwd,
               fwd_sig=None,
               bak=None,
               bak_as_function=False,
               device=None,
               use_tf_function=None):
    """Constructor.

    Args:
      fwd: A callable xs: Nested Structure -> ys: Nested Structure. Used to
        build the TensorFlow graph function that this object encapsulates.
      fwd_sig: A Nested Structure of tf.TensorSpec representing the input
        signature of `fwd`, or None (meaning that `fwd` takes no inputs). The
        actual inputs should be compatible with this (have same shapes and
        dtypes).
      bak: A callable xs, ys, dys: Nested Structure -> dxs[, dcapture]: Nested
        Structure. The custom backprop function for `fwd`. bak needs to return
        dcapture if `fwd` uses any implicitly captured tensors, whose gradients
        are dcapture.
      bak_as_function: Whether to create a TF graph function for `bak`.
      device: The device on which to run `fwd` and `bak`. Defaults to the
        current device.
      use_tf_function: Whether use tf.function. Defaults to _UseTfFunction().
    """
    self._fwd_sig = fwd_sig

    wrapped_fwd_sig = fwd_sig
    fwd_fn = fwd
    bak_fn = bak

    graph_random_seed = None
    if tf.get_default_graph().seed is not None:
      graph_random_seed = tf.get_default_graph().seed

    # Wrap the forward function to propagate framework tensors like step_seed
    # and global_step.
    wrapped_fwd_sig = NestedMap()
    self._added_global_step = False
    if GetGlobalStep() is not None:
      wrapped_fwd_sig[_FRAMEWORK_TENSOR_GLOBAL_STEP] = (
          tf.TensorSpec([], tf.int64))
      self._added_global_step = True
    if fwd_sig is not None:
      wrapped_fwd_sig.inputs = fwd_sig
    elif not wrapped_fwd_sig:
      wrapped_fwd_sig = None

    def ForwardWrapped(wrapped_inputs=None):
      if graph_random_seed is not None:
        tf.random.set_seed(graph_random_seed)
      global_step = None
      if wrapped_inputs:
        assert isinstance(wrapped_inputs, NestedMap)
        global_step = wrapped_inputs.get(_FRAMEWORK_TENSOR_GLOBAL_STEP, None)
      with GlobalStepContext(global_step):
        if wrapped_inputs and 'inputs' in wrapped_inputs:
          result = fwd(wrapped_inputs.inputs)
        else:
          result = fwd()
      return result

    fwd_fn = ForwardWrapped

    if bak:

      # Wrap the backward function to return zero gradients for framework
      # tensors like step_seed and global_step.
      def BackwardWrapped(wrapped_xs, ys, dys):
        if graph_random_seed is not None:
          tf.random.set_seed(graph_random_seed)
        with GlobalStepContext(
            wrapped_xs.get(_FRAMEWORK_TENSOR_GLOBAL_STEP, None)):
          result = bak(wrapped_xs.inputs, ys, dys)
        dxs = Transform(tf.zeros_like, wrapped_xs)
        if isinstance(result, tuple) and len(result) == 2:
          dxs.inputs, dcapture = result
          return dxs, dcapture
        else:
          dxs.inputs = result
          return dxs

      bak_fn = BackwardWrapped

    if use_tf_function is None:
      use_tf_function = _UseTfFunction()
    fn = _DefineFunction if use_tf_function else _DefineDefun
    self._data = fn(
        fwd=fwd_fn,
        fwd_sig=wrapped_fwd_sig,
        bak=bak_fn,
        bak_as_function=bak_as_function,
        device=device)

  def __call__(self, args=None):
    """Invokes the graph function.

    Args:
      args: the inputs to the graph function, must be compatible with `fwd_sig`.

    Returns:
      The output tensors with the same structure as the output of `fwd`,
      returned by a call to the graph function.
    """
    assert IsCompatible(args,
                        self._fwd_sig), '{} vs {}'.format(args, self._fwd_sig)
    return self._data.call(self.AddFrameworkInputs(args))

  @property
  def func(self):
    """The underlying TensorFlow graph function that this object encapsulates.

    The returned graph function is created by tracing `fwd` during construction.
    If not None, it will be a _DefinedFunction or ConcreteFunction that takes
    flat inputs and returns flat outputs, and can be used by routines that
    require a TensorFlow function object (e.g. tf.If, tf.While, etc).

    If no backprop function is provided during construction, the result is
    always not None.
    """
    return self._data.func

  def AddFrameworkInputs(self, inputs):
    """Add framework tensors like step_seed and global_step to inputs.

    This is only necessary when using `func`, as wrapping is handled
    automatically in __call__.

    Args:
      inputs: inputs to the function.

    Returns:
      Inputs wrapped with framework tensors suitable for use with `func`.
    """
    result = NestedMap()
    if self._added_global_step:
      global_step = GetGlobalStep()
      assert global_step is not None
      result[_FRAMEWORK_TENSOR_GLOBAL_STEP] = tf.cast(global_step, tf.int64)
    if inputs is not None:
      result.inputs = inputs
    return result if result else None

  @property
  def output_dtypes(self):
    """Output dtypes of the graph function.

    The result will have the same structure as the outputs of `fwd` but contain
    the corresponding output dtypes.
    """
    return Transform(lambda x: x.dtype, self._data.outputs)

  @property
  def stateful_ops(self):
    """Stateful ops used by `fwd`, as a list of (op_name, op_type) tuples."""
    return self._data.stateful_ops

  @property
  def captured_inputs(self):
    """Implicit input tensors captured by `fwd`."""
    return self._data.captured_inputs


def CallDefun(fwd, args=None, bak=None, bak_as_function=False, device=None):
  """Wraps fwd in a defun with custom gradient bak and calls it with args.

  Args:
    fwd: A callable xs: Nested Structure -> ys: Nested Structure.
    args: A Nested Structure of tf.Tensor or None.
    bak: A callable xs, ys, dys: Nested Structure -> dxs[, dcapture]: Nested
      Structure. The custom backprop function for fwd. bak needs to return
      dcapture if fwd uses any implicitly captured tensors, whose gradients are
      dcapture.
    bak_as_function: Whether to create a TF graph function for bak.
    device: the device on which to run fwd and bak.

  Returns:
    A Nested Structure equivalent to what fwd(args) computes.
  """
  if args is not None:
    args = Transform(tf.convert_to_tensor, args)
  sigs = Function(
      fwd_sig=TensorSpecs(args),
      bak=bak,
      bak_as_function=bak_as_function,
      device=device)(
          fwd=fwd)
  if args is None:
    return sigs()
  else:
    return sigs(args)


def If(cond, inputs, then_branch, else_branch):
  """Helper to construct an if/else statement.

  Args:
    cond: A scalar `Tensor` that can be converted to boolean.
    inputs: A flattenable representing the input tensors of the if/else
      statement. Can be None to represent no inputs.
    then_branch: A callable 'inputs' -> flattenable. The returned value should
      be compatible with what 'else_branch' returns.
    else_branch: A callable 'inputs' -> flattenable. The returned value should
      be compatible with what 'then_branch' returns.

  Returns:
    Output returned by the call to either 'then_branch' or 'else_branch'.
  """
  fwd_sig = TensorSpecs(inputs)
  then_sigs = Function(fwd_sig=fwd_sig)(fwd=then_branch)
  else_sigs = Function(fwd_sig=fwd_sig)(fwd=else_branch)
  assert IsCompatible(then_sigs.output_dtypes, else_sigs.output_dtypes), (
      'Outputs of then_branch and else_branch are not compatible: {} vs {}'
      .format(then_sigs.output_dtypes, else_sigs.output_dtypes))
  if then_sigs.captured_inputs != else_sigs.captured_inputs:
    raise ValueError('Differing captured inputs in then and else. '
                     'Ensure the same tensors are captured in the same order.')

  ret = tf.If(
      cond=cond,
      inputs=Flatten(then_sigs.AddFrameworkInputs(inputs)) +
      then_sigs.captured_inputs,
      then_branch=then_sigs.func,
      else_branch=else_sigs.func)
  return Pack(then_sigs.output_dtypes, ret)


def _Itype():
  """Loop iterator data type."""
  return tf.int32 if use_xla() else tf.int64


def WhileLoop(cond, body, loop_state):
  """Helper to construct a while loop.

  Args:
    cond: A callable NestedMap -> tf.bool.
    body: A callable NestedMap -> NestedMap.
    loop_state: A flattenable (NestedMap, list, tuple, etc.) representing the
      loop state.

  Returns:
    The final loop state in the same structure as loop_state.
  """
  fwd_sig = TensorSpecs(loop_state)
  cond_sigs = Function(fwd_sig=fwd_sig)(fwd=cond)

  def BodyWrapped(loop_state):
    result = body(loop_state)
    # loop_state is augmented with global tensors inside of DefinedFunction.
    # WhileLoop needs to return the same structure as the inputs, so we augment
    # the return value here to match.
    result = cond_sigs.AddFrameworkInputs(result)
    return result

  body_sigs = Function(fwd_sig=fwd_sig)(fwd=BodyWrapped)
  wrapped_inputs = body_sigs.AddFrameworkInputs(loop_state)
  new_state = tf.While(
      Flatten(wrapped_inputs), cond=cond_sigs.func, body=body_sigs.func)
  return Pack(wrapped_inputs, new_state).inputs


def ForLoop(body, start, limit, delta, loop_state):
  """Helper to construct a for loop.

  Args:
    body: A callable (tf.int, NestedMap) -> NestedMap.
    start: Loop variable's initial value.
    limit: Loop variable's limit value.
    delta: Loop variable's change per iteration.
    loop_state: A flattenable (NestedMap, list, tuple, etc.) representing the
      loop state.

  Returns:
    The final loop state in the same structure as loop_state.
  """
  state = NestedMap(
      iter=tf.cast(start, _Itype()),
      limit=tf.cast(limit, _Itype()),
      delta=tf.cast(delta, _Itype()),
      loop_state=loop_state)

  def LoopCond(state):
    return tf.less(state.iter, state.limit)

  def LoopBody(state):
    state.loop_state = body(state.iter, state.loop_state)
    state.iter = tf.add(state.iter, state.delta)
    return state

  return WhileLoop(LoopCond, LoopBody, state).loop_state


def TopK(x_in, k):
  """Equivalent to tf.math.top_k(x_in, k) but more efficient on tpu."""
  assert k <= 2, 'This implementation is only efficient for small k.'
  # TODO(yonghui): Try out an alternative idea where we first reshape x_in as a
  # 2d tensor, then call tf.math.top_k, and then reshape back.
  x_in_shape = x_in.shape
  x_rank = x_in_shape.rank
  assert x_rank and x_in_shape.as_list()[x_rank - 1] > 0
  last_dim_size = x_in_shape.as_list()[x_rank - 1]
  min_value = tf.math.reduce_min(x_in) - 1.0

  out_indices = []
  out_values = []

  for unused_i in range(k):
    index_i = tf.math.argmax(x_in, axis=-1, output_type=tf.int32)
    mask_i = tf.one_hot(index_i, last_dim_size)
    # TODO(yonghui): Would tf.gather be more efficient and numerically stable
    # here?
    value_i = tf.reduce_sum(mask_i * x_in, -1, keepdims=True)
    x_in = (1.0 - mask_i) * x_in + mask_i * min_value
    out_indices.append(tf.expand_dims(index_i, -1))
    out_values.append(value_i)

  if k == 1:
    return out_values[0], out_indices[0]
  else:
    return tf.concat(out_values, x_rank - 1), tf.concat(out_indices, x_rank - 1)


def ReadVariable(var_op):
  """Returns the value of the given variable operation.

  Args:
    var_op: the `Operation` object for a VarHandleOp.

  Raises:
    TypeError: if var_op is not a VarHandleOp.

  Returns:
    A `Tensor` containing the value of the variable.
  """
  if var_op.type != 'VarHandleOp':
    raise TypeError('var_op should be a VarHandleOp, got %s' % str(var_op.type))
  # Filter out the ReadVariableOps that have control dependencies to avoid
  # side-effects when the user runs it.
  filter_fn = lambda op: op.type == 'ReadVariableOp' and not op.control_inputs
  var_readers = list(filter(filter_fn, var_op.outputs[0].consumers()))
  assert var_readers
  return var_readers[0].outputs[0]


_TPU_SUMMARY_TENSORS_KEY = ('__lingvo_tpu_summary_tensors')

_get_tpu_summary_tensors = _CollectionGetter(_TPU_SUMMARY_TENSORS_KEY,
                                             lambda: [])


def AddTpuSummaryTensor(name, value, weight=1.0):
  """Adds tensor to global collection of summaries.

  This needs to be used in situations where tf.summary() could be used but
  currently tf.summary is not supported. Use py_utils.AddTpuSummaryTensor() in
  low level code to add summary tensors to global collection of summaries.
  Then recover all summary tensors from global collection by calling
  py_utils.GetTpuSummaryTensors() from top level code (for example from
  ComputeLoss method of BaseTask).

  In addition to 'name' argument, current tensorflow name scope is also
  captured and added to the metric name. This way for example summaries from
  a repeated layer will appear as separate graphs in the tensorboard.

  Weight argument is optional and defaults to 1.0. See BaseTask.ComputeLoss for
  the exact definition of weight for eval metrics.

  Args:
    name: metric name
    value: metric value tensor
    weight: weight tensor for weighted metrics
  """
  tpu_summary_tensors = _get_tpu_summary_tensors()
  x = NestedMap()
  x.name = name
  x.value = value, tf.convert_to_tensor(weight)
  x.name_scope = tf.get_default_graph().get_name_scope()
  tpu_summary_tensors.append(x)


def GetTpuSummaryTensors():
  """Returns summary tensors from global collection.

  Returns:
    A dict containing str keys and (metric, weight) pairs as values
  """
  tpu_summary_tensors = _get_tpu_summary_tensors()
  return {
      '%s/%s' % (x.name, SanitizeScopeKey(x.name_scope)): x.value
      for x in tpu_summary_tensors
  }


def ClearTpuSummaryTensors():
  tpu_summary_tensors = _get_tpu_summary_tensors()
  del tpu_summary_tensors[:]


def ComputationShape(split_size, topology=None):
  """Decides the computation shape based on the split_size.

  Args:
    split_size: number of accelerators to use per split.
    topology: a serialized string of `tensorflow.tpu.TopologyProto`, or a
      `tf.tpu.experimental.Topology` object, that describes the TPU cluster
      topology. If not set, it'll use a default setting based on split_size.

  Returns:
    A 4-element list that describes the computation shape.
  """
  if topology:
    if isinstance(topology, tf.tpu.experimental.Topology):
      topology_info = topology
    else:
      topology_info = tf_topology.Topology(serialized=topology)
  computation_shape = None
  if topology and functools.reduce(lambda a, b: a * b,
                                   topology_info.mesh_shape) == split_size:
    computation_shape = topology_info.mesh_shape
  elif split_size == 1:
    computation_shape = [1, 1, 1, 1]
  elif topology and topology_info.mesh_shape[
      -1] == 1 and split_size in topology_info.mesh_shape:
    # For Megacore, if we find exact match on mesh shape, map split_size to it
    computation_shape = [1, 1, 1, 1]
    computation_shape[topology_info.mesh_shape.tolist().index(
        split_size)] = split_size
  elif split_size == 2:
    computation_shape = [1, 1, 1, 2]
  elif split_size == 4:
    computation_shape = [1, 2, 1, 2]
  elif split_size == 8:
    computation_shape = [2, 2, 1, 2]
  elif split_size == 16:
    computation_shape = [4, 2, 1, 2]
  elif split_size == 32:
    computation_shape = [4, 4, 1, 2]
  elif split_size == 64:
    if topology and topology_info.mesh_shape[1] == 32:
      # Fwd within-replica all-reduces is performed along column;
      # Bwd gradient cross-replica all-reduces is performed along row.
      # This currently has better performance than the strided patten.
      computation_shape = [1, 32, 1, 2]
    else:
      computation_shape = [4, 8, 1, 2]
  elif split_size == 128:
    computation_shape = [8, 8, 1, 2]
  elif split_size == 256:
    computation_shape = [8, 16, 1, 2]
  elif split_size == 512:
    computation_shape = [16, 16, 1, 2]
  elif split_size == 1024:
    computation_shape = [16, 32, 1, 2]
  elif split_size == 2048:
    computation_shape = [32, 32, 1, 2]
  elif split_size == 4096:
    computation_shape = [64, 32, 1, 2]
  elif split_size == 8192:
    computation_shape = [128, 32, 1, 2]
  else:
    assert False, ('Model parallelism with %d devices is currently not'
                   ' supported.' % split_size)
  assert computation_shape is not None
  return computation_shape


def GetExtraVars():
  """Returns the captured variables by the function."""
  g = tf.get_default_graph()
  if isinstance(g, func_graph.FuncGraph):
    return g.variable_captures
  return function.get_extra_vars()


def GetExtraInputs():
  """Returns the captured input tensors by the function."""
  g = tf.get_default_graph()
  if isinstance(g, func_graph.FuncGraph):
    return g.external_captures
  return function.get_extra_inputs()


def GetExtraArgs():
  """Returns the corresponding function arguments for the captured inputs."""
  g = tf.get_default_graph()
  if isinstance(g, func_graph.FuncGraph):
    return g.internal_captures
  return function.get_extra_args()


def ShardedFilePatternToGlob(file_pattern):
  """Converts a file pattern path@shards to path-?????-of-shards."""
  if ',' in file_pattern:
    raise ValueError(
        'ShardedFilePatternToGlob does not support multiple file patterns.')
  if '@' not in file_pattern:
    return file_pattern
  path, shards = file_pattern.split('@')
  if shards == '*':
    return f'{path}-?????-of-*'
  return f'{path}-?????-of-{int(shards):05}'


def ComputeNceAndAuc(probs, targets, mask):
  """Compute normalized cross entropy and AUC of the PR curve for a batch.

  Args:
    probs: a tensor of shape [batch, time].
    targets: a tensor of shape [batch, time], where each element is either 0 or
      1 indicating wrong or correct.
    mask: a tensor of shape [batch, time], a mask for hyp sequence.

  Returns:
    nce: a tensor of shape [1], the normalized cross entropy value.
    auc: a tensor of shape [1], the AUC value.
  """

  def LogWithClip(tensor, clip_value_min=1e-8):
    """Clip all elements of a tensor to a minimum before taking log."""
    return tf.math.log(tf.clip_by_value(tensor, clip_value_min, 1.0))

  bce = -targets * LogWithClip(probs) - (1 - targets) * LogWithClip(1 - probs)
  num_cor = tf.reduce_sum(targets * mask)
  num_tokens = tf.reduce_sum(mask)
  wcr = num_cor / num_tokens
  entropy = -wcr * LogWithClip(wcr) - (1 - wcr) * LogWithClip(1 - wcr)
  avg_conditional_entropy = tf.reduce_mean(tf.boolean_mask(bce, mask))
  nce = (entropy - avg_conditional_entropy) / entropy
  auc = tf.metrics.auc(targets, probs, mask, curve='PR')[1]
  return nce, auc


def GatherTensorValuesBySeqIndices(tensor, class_indices, keepdims=False):
  """Gather values from a 3d tensor according to sequences of indices.

  Args:
    tensor: a 3d tensor of [dim0, dim1, num_class], e.g. output from softmax.
    class_indices: a 2d tensor of [dim0, dim1], where the second dim is a
      sequence of class indices between 0 to num_class - 1, inclusive.
    keepdims: bool, expand the last dimension of the returned tensor if True.

  Returns:
    A tensor ret of [dim0, dim1], where
      ret[b, t] = tensor[b, t, indices[b, t]].
      If keepdims is True, then ret has shape [dim0, dim1, 1].
  """
  tensor = HasRank(tensor, 3)
  class_indices = HasRank(class_indices, 2)
  tensor = HasShape(tensor, GetShape(class_indices), 2)
  dim0 = GetShape(class_indices)[0]
  dim1 = GetShape(class_indices)[1]
  dim0_indices = tf.tile(tf.expand_dims(tf.range(dim0), axis=-1), [1, dim1])
  dim1_indices = tf.tile(tf.expand_dims(tf.range(dim1), axis=0), [dim0, 1])
  gather_indices = tf.stack([
      tf.cast(dim0_indices, dtype=class_indices.dtype),
      tf.cast(dim1_indices, dtype=class_indices.dtype), class_indices
  ],
                            axis=-1)
  ret = tf.gather_nd(tensor, gather_indices)
  if keepdims:
    ret = tf.expand_dims(ret, axis=-1)
  return ret


def GetSoftmaxProbsBySeqIndices(logits, indices, keepdims=False):
  """Get softmax probabilities from index sequences given logits sequences.

  Args:
    logits: a tensor of [batch, time, num_class] or [time, batch, num_class].
    indices: a tensor of [batch, time] or [time, batch].
    keepdims: bool, expand the last dimension of the returned tensor if True.

  Returns:
    a tensor of [batch, time] or [time, batch] for the corresponding softmax
      probabilities. If keepdims is True, returned tensor has a third dimension
      of size 1.
  """
  probs = tf.nn.softmax(logits)
  return GatherTensorValuesBySeqIndices(probs, indices, keepdims)


def DivideNoNan(x, y):
  """Equivalent to tf.math.divide_no_nan but supports bfloat16."""
  safe_y = tf.where(tf.equal(y, 0.), tf.ones_like(y), y)
  return tf.where(tf.equal(y, 0.0), tf.zeros_like(x), x / safe_y)


def SequencePaddings(seqlen, maxlen=None):
  mask = tf.sequence_mask(seqlen, maxlen, dtype=tf.float32)
  return 1 - mask


def AppendDims(x, ndims):
  return tf.reshape(x, GetShape(x) + [1] * ndims)
