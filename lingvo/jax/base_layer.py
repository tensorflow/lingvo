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
"""Base class for the lingvo Jax layers."""

import contextlib
import copy
import enum
import hashlib
import itertools
import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from absl import flags
from absl import logging
import jax
from jax import numpy as jnp
from jax import random as jrandom
from jax.experimental import pjit
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit
WeightParams = py_utils.WeightParams
IsDefaultParamInit = py_utils.IsDefaultParamInit
DefaultParamInit = py_utils.DefaultParamInit

ParamsT = pytypes.ParamsT
InstantiableParams = py_utils.InstantiableParams
BaseLayerS = TypeVar('BaseLayerS', bound='BaseLayer')
BaseLayerT = TypeVar('BaseLayerT', bound='BaseLayer')
BaseLayerParamsT = InstantiableParams[BaseLayerT]
JTensor = pytypes.JTensor
PRNGKey = pytypes.PRNGKey
JTensorOrPartitionSpec = pytypes.JTensorOrPartitionSpec
NpTensor = pytypes.NpTensor
SummaryDict = pytypes.SummaryDict

NestedJTensor = pytypes.NestedJTensor
NestedBool = pytypes.NestedBool
NestedParams = pytypes.NestedParams
NestedJTensorOrPartitionSpec = pytypes.NestedJTensorOrPartitionSpec

SplitDimsMapping = pytypes.SplitDimsMapping

# Layer stack to establish parent child relationships.
_LAYER_STACK = py_utils.ThreadLocalStack()

# Global state that may impact how certain jax computation will be carried (e.g.
# whether or not to enable dropout).
_JaxContextStack = py_utils.ThreadLocalStack()

# A few special variable collections.
# Variable collections annotates variables with special properties, e.g. whether
# or not the variable is leanable, whether or not the variable is subject to lp
# regularization.
SKIP_LP_REGULARIZATION = '__lingvo_jax_skip_regularization'
NON_TRAINABLE = '_lingvo_jax_non_trainable'
FLAX_VARIABLE = '_flax_variable'
REQUIRES_MEAN_SYNC = '_requires_mean_sync'


def VarNotTrainable(var_params: ParamsT) -> bool:
  """Returns True if var_params is not a trainable variable."""
  return NON_TRAINABLE in var_params.collections


def VarRequiresMeanSync(var_params: ParamsT) -> bool:
  """Returns True if var_params requires synchronization across replicas."""
  return REQUIRES_MEAN_SYNC in var_params.collections


def ToPartitionSpec(split_dims_mapping: SplitDimsMapping,
                    mesh_axis_names: Sequence[str]) -> pjit.PartitionSpec:
  """Converts split_dims_mapping to pjit.PartitionSpec.

  Args:
    split_dims_mapping: A (nested) tuple of mesh axis to split x over. Below are
      a few example sharding specifications. (0, 2) - in this case, the first
      dim of x is split over the first axis of the mesh and the second dim over
      the third axis of the mesh. (1, -1) - in this case, the first dim of x is
      split over the second axis of the mesh and the second dim is replicated.
      (1, None) - in this case the first dim is split over the second axis of
      the mesh, and the second dim replicated. ('data', 'mdl') - in this case,
      the first dim is split over the 'data' axis of the mesh and the second dim
      over the 'mdl' axis. (('replica', 'data'), 'mdl'), in this case the first
      dim is split over both the 'replica' and 'data' axes, while the second dim
      over the 'mdl' axis.
    mesh_axis_names: A tuple/list of strings of the name of the device mesh.

  Returns:
    A pjit.PartitionSpec.
  """

  def _ParseSplitDims(dims_mapping):
    split_dims = []

    for s_i in dims_mapping:
      if isinstance(s_i, int):
        if s_i < 0:
          split_dims.append(None)
        else:
          assert s_i < len(mesh_axis_names)
          split_dims.append(mesh_axis_names[s_i])
      elif isinstance(s_i, str):
        assert s_i in mesh_axis_names
        split_dims.append(s_i)
      elif isinstance(s_i, (tuple, list)):
        split_dims.append(_ParseSplitDims(s_i))
      else:
        assert s_i is None
        split_dims.append(None)

    return tuple(split_dims)

  partition_spec = _ParseSplitDims(split_dims_mapping)
  return pjit.PartitionSpec(*partition_spec)


def VarPartitionSpecs(
    var_specs: NestedParams, device_mesh: NpTensor,
    device_axis_names: List[str]) -> NestedJTensorOrPartitionSpec:
  """Given variable specs (WeightParams), returns pjit partition specs.

  Args:
    var_specs: A nested structure of variable weight params (created via
      py_utils.WeightParams).
    device_mesh: A numpy array of device mesh.
    device_axis_names: Axis name for each mesh axis.

  Returns:
    A nested structure of PartitionSpec.
  """

  assert len(device_axis_names) == len(device_mesh.shape)

  def _GetSpec(var_p):
    v_shape = var_p.shape
    # v_split_dim_mapping may contain a mixture of -1, integers, str, or None.
    # -1 and None both indicates that the corresponding dim is not partitioned.
    v_split_dim_mapping = var_p.tensor_split_dims_mapping
    if v_split_dim_mapping is not None:
      assert len(v_split_dim_mapping) == len(v_shape)
    else:
      v_split_dim_mapping = [-1] * len(v_shape)

    if var_p.repeat_prefix is not None:
      repeat_prefix = var_p.repeat_prefix
      if var_p.repeat_prefix_split_dims_mapping is not None:
        prefix_split_dims_mapping = var_p.repeat_prefix_split_dims_mapping
        assert len(prefix_split_dims_mapping) == len(repeat_prefix)
      else:
        prefix_split_dims_mapping = [-1] * len(repeat_prefix)
      # Append sharding annotations for the prefix part.
      v_split_dim_mapping = (
          list(prefix_split_dims_mapping) + list(v_split_dim_mapping))

    return ToPartitionSpec(v_split_dim_mapping, device_axis_names)

  return jax.tree_map(_GetSpec, var_specs)


def GlobalMeshDefined() -> bool:
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = jax.experimental.maps.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


# This wrapped with_sharding_constraint will not throw error for eval_shape
# outside pjit. It is also used in p5x.
def WithShardingConstraint(
    x: JTensor, axis_resources: Optional[pjit.PartitionSpec]) -> JTensor:
  """Wrapper for pjit with_sharding_constraint, no-op on cpu or outside pjit."""
  if jax.devices()[0].platform == 'cpu' or not GlobalMeshDefined():
    return x
  else:
    return pjit.with_sharding_constraint(x, axis_resources)


_DF_DEVICE_KIND = 'TPU v3'
_PF_DEVICE_KIND = 'TPU v4'


def CreateDeviceMesh(mesh_shape: Tuple[int, int, int]) -> np.ndarray:
  """Creates a device mesh."""

  def MaybeReorderMesh(mesh_shape: Tuple[int, int, int],
                       device_mesh: np.ndarray) -> np.ndarray:
    """Possibly re-orders device mesh for better performance."""
    device_kind = jax.devices()[0].device_kind
    logging.info('device_kind: "%s", mesh_shape: %s', device_kind, mesh_shape)

    if device_kind == _DF_DEVICE_KIND and mesh_shape[-1] == 8:
      logging.info('Re-order DF device mesh for better performance.')
      perm = np.array([0, 1, 2, 3, 6, 7, 4, 5])
      device_mesh = device_mesh[:, :, perm]
    elif device_kind == _PF_DEVICE_KIND and mesh_shape == (1, 16, 4):
      logging.info('Re-order PF device mesh for better performance.')
      perm = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
      device_mesh = device_mesh.reshape((4, 16))[:, perm].reshape(1, 16, 4)
    elif device_kind == _PF_DEVICE_KIND and mesh_shape == (1, 32, 8):
      logging.info('Re-order PF device mesh for better performance.')
      perm = np.array([0, 1, 4, 5, 2, 3, 6, 7])
      # host-untiling
      device_mesh = device_mesh.reshape((-1, 8))[:, perm]
      # x (minor), z(major) for 32, and y for 8
      device_mesh = device_mesh.reshape((8, 8, 4)).transpose(
          (0, 2, 1)).reshape(mesh_shape)
    elif device_kind == _PF_DEVICE_KIND and mesh_shape == (1, 64, 8):
      logging.info('Re-order PF device mesh for better performance.')
      perm = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15])
      device_mesh = device_mesh.reshape((-1, 16))[:, perm].reshape(1, 64, 8)
    elif device_kind == _PF_DEVICE_KIND and mesh_shape == (1, 64, 16):
      logging.info('Re-order PF device mesh for better performance.')
      device_mesh = device_mesh.reshape((16, 64, 1)).transpose()
    return device_mesh

  device_mesh = np.asarray(jax.devices()).reshape(mesh_shape)
  return MaybeReorderMesh(mesh_shape, device_mesh)


def MaybeShard(x: JTensor,
               split_dims_mapping: Optional[SplitDimsMapping] = None,
               mesh_axis_names: Optional[Sequence[str]] = None) -> JTensor:
  """Adds explicit xla sharding constraints.

  This is a wrap around jax.with_sharding_constraint to allow for adding
  explicit sharding annotations to an intermediate node in a jax function.

  No sharding annotation is added if either split_dims_mapping is None or
  mesh_axis_names is None.

  Args:
    x: the input tensor to be sharded.
    split_dims_mapping: A (nested) tuple of mesh axis to split x over. Below are
      a few example sharding specifications. (0, 2) - in this case, the first
      dim of x is split over the first axis of the mesh and the second dim over
      the third axis of the mesh. (1, -1) - in this case, the first dim of x is
      split over the second axis of the mesh and the second dim is replicated.
      (1, None) - First dim is split over the second dim of the mesh, and the
      second dim replicated. ('data', 'mdl') - in this case,  the first dim is
      split over the 'data' axis of the mesh and the second dim over the 'mdl'
      axis. (('replica', 'data'), 'mdl'), in this case the first dim is split
      over both the 'replica' and 'data' axes, while the second dim over the
      'mdl' axis.
    mesh_axis_names: A tuple/list of strings of the name of the device mesh.

  Returns:
    An annotated JTensor.
  """
  if split_dims_mapping is None or mesh_axis_names is None:
    return x

  assert len(x.shape) == len(split_dims_mapping), (
      f'Invalid split_dims_mapping. Expected len(split_dims_mapping) '
      f'is {len(x.shape)}, while it is {len(split_dims_mapping)}. '
      f'x.shape = {x.shape} and split_dims_mapping = {split_dims_mapping}')
  partition_spec = ToPartitionSpec(split_dims_mapping, mesh_axis_names)
  return WithShardingConstraint(x, partition_spec)


def GenerateSeedFromName(name: str) -> np.int64:
  """Generates a random seed from a name string.

  Args:
    name: A string.

  Returns:
    An integer seed in the range [0, 2**31 - 1).
  """
  md5 = hashlib.md5()
  md5.update(name.encode('utf-8'))
  return np.int64(int(md5.hexdigest(), 16) % (2**31 - 1))


def GetFanInFanOut(shape: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
  """Returns (fan_in, fan_out) of a weight variable of the given shape."""
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


def InitVar(var_full_name: str, var_p: ParamsT, prng_key: PRNGKey) -> JTensor:
  """Creates an initial value of a var."""
  method = var_p.init.method
  scale = var_p.init.scale
  assert isinstance(scale, float)
  shape = var_p.shape
  # Final_shape can be different from shape if this var should be repeated. All
  # the stats of this variable should be computed based on the original shape.
  if var_p.repeat_prefix is not None:
    assert all([prefix_size > 0 for prefix_size in var_p.repeat_prefix])
    final_shape = list(var_p.repeat_prefix) + list(shape)
  else:
    final_shape = shape
  init_dtype = var_p.dtype
  if shape:
    assert all([dim_size > 0 for dim_size in shape]), shape
    dim0 = shape[0]
  else:
    dim0 = 1

  if IsDefaultParamInit(var_p.init):
    logging.warning(
        'WARNING!!! var %s is using the default xavier initializer.'
        ' Make sure this is intended.', var_full_name)

  if (method in [
      'gaussian_sqrt_dim', 'uniform_sqrt_dim', 'truncated_gaussian_sqrt_dim'
  ]):
    if len(shape) > 2:
      # This is probably not the right method to use when len(shape) > 2,
      # e.g. dim0 will be 3 with a 3x3 conv2d kernel.
      logging.warning(
          'Initializing %s of shape %s with method %s: dim0=%s. '
          'Make sure that it is intended.', var_full_name, shape, method, dim0)
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

  name_hash = GenerateSeedFromName(var_full_name)
  prng_key = jax.random.fold_in(prng_key, name_hash)

  if method in [
      'gaussian', 'gaussian_sqrt_dim', 'gaussian_sqrt_fanin',
      'gaussian_sqrt_fanout', 'gaussian_sqrt_fanavg'
  ]:
    return scale * jrandom.normal(prng_key, final_shape, init_dtype)
  elif method in ['uniform', 'uniform_sqrt_dim']:
    return scale * jrandom.uniform(
        prng_key, final_shape, init_dtype, minval=-1.0, maxval=1.0)
  elif method in [
      'truncated_gaussian', 'truncated_gaussian_sqrt_dim',
      'truncated_gaussian_sqrt_fanin', 'truncated_gaussian_sqrt_fanout'
  ]:
    return scale * jrandom.truncated_normal(
        prng_key, lower=-2.0, upper=2.0, shape=final_shape, dtype=init_dtype)
  elif method in ['constant']:
    return scale + jnp.zeros(shape=final_shape, dtype=init_dtype)
  elif method in ['xavier']:
    fan_in, fan_out = GetFanInFanOut(shape)
    limit = scale * math.sqrt(6. / (fan_in + fan_out))
    return limit * jrandom.uniform(
        prng_key, final_shape, init_dtype, minval=-1.0, maxval=1.0)
  else:
    assert False, 'init_type %s not supported.' % method


class _PrngKey:
  """Random number generator keys."""

  def __init__(self) -> None:
    self._prng_key: JTensor = None
    self._global_step: JTensor = None

  def ResetKey(self, prng_key: JTensor, global_step: JTensor) -> None:
    self._prng_key = prng_key
    self._global_step = global_step

  def ClearKey(self) -> None:
    self._prng_key = None
    self._global_step = None

  @property
  def global_step(self) -> JTensor:
    assert self._global_step is not None
    return self._global_step

  def NextKey(self) -> JTensor:
    assert self._prng_key is not None
    self._prng_key, next_key = jrandom.split(self._prng_key)
    return next_key


class _SummaryDict:
  """A dict holding summaries generated during forward computation."""

  def __init__(self) -> None:
    self.dict = {}

  def AddSummary(self, name: str, tensor: JTensor) -> None:
    """Adds named summary to the thread local dict."""
    prefix = '/'.join(_NAMESPACE_STACK.stack)
    summary_name = prefix + '/' + name
    next_iter = 0
    full_name = summary_name
    while full_name in self.dict:
      next_iter += 1
      full_name = summary_name + str(next_iter)
    self.dict[full_name] = tensor

  def Clear(self) -> None:
    """Clears all summaries."""
    self.dict = {}


class JaxContext:
  """Global context under which jax computations are carried out."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for `JaxContent`."""
    p = InstantiableParams(cls)
    p.Define('do_eval', None, 'Whether to do eval.')
    p.Define('in_unit_test', None, 'Whether this is running in a unit test.')
    p.Define(
        'enable_asserts', None, 'If set to non-None, '
        'this value is used instead of FLAGS.enable_asserts. '
        'If False, we disable all asserts ops and return tf.no_op() instead.')
    p.Define(
        'enable_check_numerics', None, 'If set to non-None, '
        'this value is used instead of FLAGS.enable_check_numerics. '
        'If False, we bypass calls to CheckNumerics.')
    return p

  def __init__(self, params: ParamsT) -> None:
    self._params = params.Copy()
    self._prng_key = _PrngKey()
    self._summary_dict = _SummaryDict()

  @property
  def prng_key(self) -> _PrngKey:
    return self._prng_key

  @property
  def summary_dict(self) -> _SummaryDict:
    return self._summary_dict

  @property
  def params(self) -> ParamsT:
    return self._params

  @property
  def do_eval(self) -> bool:
    return self.params.do_eval

  def __enter__(self) -> 'JaxContext':
    _JaxContextStack.stack.append(self)
    return self

  def __exit__(self, type_arg, value_arg, traceback_arg):
    assert _JaxContextStack.stack
    assert _JaxContextStack.stack[-1] is self
    _JaxContextStack.stack.pop()

  @staticmethod
  def Top() -> Optional['JaxContext']:
    return _JaxContextStack.stack[-1] if _JaxContextStack.stack else None

  @staticmethod
  def NewContext(*,
                 params: Optional[ParamsT] = None,
                 prng_key: Optional[JTensor] = None,
                 global_step: Optional[JTensor] = None) -> 'JaxContext':
    """Returns a new empty JaxContext.

    Args:
      params: if not None, and instance of JaxContext.Params(). If it is None,
        the newly constructed JaxContext will assume the same params as the
        current context if it is not None, or the default one.
      prng_key: If not None, the new root prng_key
      global_step: If not None, the global step for the context. Note, if
        prng_key is not None, global_step can't be None either.

    Returns:
      A new JaxContext.
    """
    if params is None:
      current = JaxContext.Top()
      if current is None:
        new_params = JaxContext.Params()
      else:
        new_params = current.params.Copy()
    else:
      new_params = params.Copy()
    context = JaxContext(new_params)
    if prng_key is not None:
      assert global_step is not None
      context.prng_key.ResetKey(prng_key, global_step)
    return context


def CurJaxContext() -> JaxContext:
  current = JaxContext.Top()
  assert current is not None
  return current


def AddSummary(name: str, tensor: JTensor) -> None:
  context = CurJaxContext()
  context.summary_dict.AddSummary(name, tensor)


def ClearSummary() -> None:
  context = CurJaxContext()
  context.summary_dict.Clear()


def AllSummaries() -> SummaryDict:
  context = CurJaxContext()
  return context.summary_dict.dict


def NextPrngKey() -> JTensor:
  context = CurJaxContext()
  return context.prng_key.NextKey()


def ResetPrngKey(prng_key: JTensor, global_step: JTensor) -> None:
  context = CurJaxContext()
  context.prng_key.ResetKey(prng_key, global_step)


def CurGlobalStep() -> JTensor:
  context = CurJaxContext()
  return context.prng_key.global_step


_NAMESPACE_STACK = py_utils.ThreadLocalStack()


@contextlib.contextmanager
def NameSpace(name: str):
  NestedMap.CheckKey(name)
  _NAMESPACE_STACK.stack.append(name)
  try:
    yield
  finally:
    _NAMESPACE_STACK.stack.pop()


def _BaseLayerInitWrapper(func):
  """A decorator for layer's __init__.

  Args:
    func: The __init__ method of `BaseLayer`'s subclasses.

  Returns:
    A decorator wrapper for layer's initializer. Note that this wrapper can
    be called multiple times for the same layer instance, once for each
    __init__() for classes on the class hierarchy.
  """

  def Wrapper(self, *args: Any, **kwargs: Any) -> None:
    """Decorator wrapper fn."""
    stack = _LAYER_STACK.stack
    if stack and stack[-1] is self:
      # Short circuit if called multiple times (eg. super() chain).
      func(self, *args, **kwargs)
      return

    # Push back self (the current layer) to the stack.
    stack_size = len(stack)
    stack.append(self)
    try:
      # Calls the layer's real __init__ method.
      func(self, *args, **kwargs)
      if len(stack) > 1:
        # Records the fact stack[-2] just created a sub-layer self.
        stack[-2]._AutoAddChild(self)  # pylint: disable=protected-access
    finally:
      # Pop out self (the current layer).
      assert stack[-1] is self
      stack.pop()
      assert len(stack) == stack_size

  return Wrapper


def _BaseLayerFuncWrapper(func, fname: str):
  """A decorator for layer's func.

  Args:
    func: A method of `BaseLayer`'s subclasses.
    fname: func name.

  Returns:
    A decorator wrapper for the method.
  """

  def Wrapper(self, *args, **kwargs):
    """Decorator wrapper fn."""
    assert isinstance(self, BaseLayer)
    lname = self.params.name
    with NameSpace(f'{lname}.{fname}'):
      return func(self, *args, **kwargs)

  return Wrapper


class CreateLayerVariablesStatus(enum.Enum):
  # Variable creation is not enabled, e.g. during layer initialization.
  NOT_ENABLED = 0
  # Variable creation is enabled, only during the InstantiateVariableConfigs()
  # call.
  ENABLED = 1
  # Variable creation has completed, no more variables can be created.
  COMPLETED = 2


class BaseLayerMeta(type):
  """Metaclass tracking child layers and variable initialization."""

  # pylint: disable=bad-mcs-classmethod-argument
  def __new__(mcs, name, bases, dct):
    cls = super(BaseLayerMeta, mcs).__new__(mcs, name, bases, dct)
    if '__init__' not in dct:

      def TrivialInit(self, params):
        super(cls, self).__init__(params)  # pylint: disable=bad-super-call

      cls.__init__ = TrivialInit

    cls.__init__ = _BaseLayerInitWrapper(cls.__init__)

    if 'FProp' in dct:
      cls.FProp = _BaseLayerFuncWrapper(cls.FProp, 'FProp')
    if 'InitStates' in dct:
      cls.InitStates = _BaseLayerFuncWrapper(cls.InitStates, 'InitStates')
    if 'ExtendStep' in dct:
      cls.ExtendStep = _BaseLayerFuncWrapper(cls.ExtendStep, 'ExtendStep')

    return cls

  # pylint: enable=bad-mcs-classmethod-argument

  def __call__(cls, *args, **kwargs):
    self = super().__call__(*args, **kwargs)
    # This happens after self.__init__()
    # pylint: disable=protected-access
    self._disable_create_child = True
    self._VerifyChildren()
    # pylint: enable=protected-access
    return self


class BaseLayer(metaclass=BaseLayerMeta):
  r"""Base class for all the layer object.

  Subclasses are expected to override the following functions:

  Params(): Returns a configuration Params for this layer.
  __init__: Initializes this layer and its sub-layers.
  CreateLayerVariables(): Register variables to be created.
  FProp(): The main method that carries out ML computation.

  Optionally, if a sub-class would like to update some params in the forward
  pass (e.g.  update batch norm moving avg and variance), one can do so by
  calling ForwardUpdateVar() to record the variable being updated and the new
  param value. User of this layer (e.g. the trainer loop) is responsible for
  fetching those updated vars. A sub-class can also record summaries via
  AddSummary() method.

  Users of this class are expected to call the following functions in sequence.
  The following is part of a train loop.

  # Instantiate a layer object.
  layer_params = XXX.Params().Set()
  layer = layer_params.Instantiate()

  # Instantiate all layer variables.
  prng_key = jrandom.PRNGKey(seed=xxx)
  prng_key, init_keys = jrandom.split(prng_key)
  initial_variables = layer.InstantiateVariables(init_key)

  # Set up random number generation key for FProp
  prng_key, fprop_key = jrandom.split(prng_key)
  global_step = jnp.array(0, dtype=jnp.unint64)

  # Fetch inputs to fed to a model.
  inputs = .... # Get some input for the model.

  # The main compute loop. This is a pure function without side effect.
  def Compute(theta, prng_key, global_step, inputs):
    with jax_base_layer.JaxContext.NewContext():
      # Mix in global seed so that rng_seed are different for different steps.
      per_step_prng_key = jrandom.fold_in(prng_key, global_step)
      jax_base_layer.ResetPrngKey(per_step_prng_key, global_step)
      # Prepare the layer and all its sub-layers for the FProp call.
      layer.PrepareFProp()
      output = layer.FProp(theta, inputs)
      # fetch params that possibly being updated during forward pass.
      forward_updated_theta = layer.forward_updated_vars

      def get_new_param(old, new):
        if new is not None:
          return new
        else:
          return old

      # Get the new variables.
      new_theta = tf.nest.map_structure(get_new_param, theta,
                                        forward_updated_theta)
      # Fetch summaries.
      summaries = jax_base_layer.AllSummaries()
      # Finally, reset the prng key.
      jax_base_layer.ClearPrngKey()
      global_step += 1

      return new_theta, global_step, output, summaries

  # Jit compute function.
  compute_jit = jax.jit(compute)

  new_theta, global_step, output, summaries = compute_jit(
      initial_variables, fprop_key, global_step, inputs)

  Note, the above code-snippet doesn't include a backward pass.
  """

  @classmethod
  def Params(cls: Type[BaseLayerT]) -> BaseLayerParamsT:
    """Returns the layer params."""
    p = InstantiableParams(cls)
    p.Define('name', '',
             'Name of this layer object, must be a valid identifier.')
    p.Define('dtype', jnp.float32, 'Default dtype for all variables.')
    # None value will make FProp use dtype instead of fprop_dtype.
    p.Define('fprop_dtype', None, 'Activations datatype to use.')
    p.Define(
        'params_init', DefaultParamInit(),
        'How model weights should be initialized. Not to be confused with '
        'hyperparams.')

    p.Define(
        'skip_lp_regularization', None,
        'If True, all variables in this layer will skip Lp regularization. '
        'If None/False, only variables explicitly in the '
        'SKIP_LP_REGULARIZATION collection will skip Lp regularization. '
        'Also propagated to child layers with default settings (None).')

    p.Define(
        'repeat_prefix', None,
        'How many times this layer should be repeated. If not None, this layer'
        ' is repeated repeat_prefix times. All variables of this layer are '
        ' repeated repeat_prefix times at construction. repeat_prefix can be'
        ' a list in case this layer is nested under multiple repeat layers.')
    # NOTE(yonghui): Should repeat_prefix_split_dims_mapping be moved to
    # p.weight_split_dims_mapping?
    p.Define(
        'repeat_prefix_split_dims_mapping', None,
        'If not None, should have the same length as repeat_prefix. This param'
        ' specifies how repeat_prefix prefix dims of variables are partitioned'
        ' over a device mesh.')

    # SPMD partition related params.
    p.Define(
        'device_mesh', None,
        'A numpy.ndarray specifying the topology of a device mesh to place the'
        ' computations onto. If device_mesh is None, it is assumed to be a'
        ' single device. Here are some examples:'
        ' np.array([0, 1, 2, 3, 4, 5, 6, 7]) which is a 1d mesh with 8 devices,'
        ' np.array([[0, 1, 2, 3], [4, 5, 6, 7]]) which is 2d matrix of 8'
        ' devices.')
    p.Define('mesh_axis_names', None, 'Names for each mesh axis.')
    p.Define(
        'weight_split_dims_mapping', py_utils.Params(),
        'Relevant only if device_mesh above is not None. It specifies how '
        'weight of this layer or those of the sublayers should be sharded '
        'over device mesh.')
    wp = p.weight_split_dims_mapping
    wp.Define(
        'wt', None,
        'The sharding annotation for the model weight. This usually '
        'refers to the primary model weight. Sub-layers can define '
        'additional params for more weights.')
    p.Define(
        'activation_split_dims_mapping', py_utils.Params(),
        'Relevant only if device_mesh above is not None. It specifies how '
        'activation of this layer or those of the sublayers should be '
        'sharded over device mesh.')
    ap = p.activation_split_dims_mapping
    ap.Define(
        'out', None, 'The sharding annotation of the activation. This usually '
        'refers to the primary layer output. Sub-layers '
        'can define additional params for more activations.')
    return p

  @staticmethod
  def CopyBaseParams(from_params: InstantiableParams[BaseLayerS],
                     to_params: BaseLayerParamsT) -> BaseLayerParamsT:
    """Copies BaseLayer params from `from_params` to `to_params`."""
    assert issubclass(from_params.cls, BaseLayer)
    assert issubclass(to_params.cls, BaseLayer)
    # Copy-over the BaseLayer params.
    if to_params.repeat_prefix is None:
      to_params.repeat_prefix = from_params.repeat_prefix
    if to_params.repeat_prefix_split_dims_mapping is None:
      to_params.repeat_prefix_split_dims_mapping = (
          from_params.repeat_prefix_split_dims_mapping)
    if to_params.dtype == jnp.float32:
      to_params.dtype = from_params.dtype
    if to_params.fprop_dtype is None:
      to_params.fprop_dtype = from_params.fprop_dtype
    if to_params.skip_lp_regularization is None:
      to_params.skip_lp_regularization = from_params.skip_lp_regularization
    if to_params.device_mesh is None:
      to_params.device_mesh = copy.deepcopy(from_params.device_mesh)
    if to_params.mesh_axis_names is None:
      to_params.mesh_axis_names = copy.deepcopy(from_params.mesh_axis_names)
    if IsDefaultParamInit(to_params.params_init):
      # Copy over params_init as well.
      to_params.params_init = from_params.params_init.Copy()
    return to_params

  def __init__(self, params: BaseLayerParamsT) -> None:
    """Layer constructor.

    Args:
      params: A params used to construct this layer.
    """
    assert params.name, ('Layer params for %s must have a "name"' %
                         self.__class__.__name__)

    module_name = params.name
    NestedMap.CheckKey(module_name)

    self._parent = None
    for parent in reversed(_LAYER_STACK.stack):
      if parent is not self:
        self._parent = parent
        break
    self._params = params.Copy()
    logging.debug('Creating layer %s with params: \n %s \n',
                  self.__class__.__name__, str(params))
    # Vars created by this layer. Frozen after the InstantiateVariables() call.
    self._private_vars = NestedMap()
    # Child layers created by this layer through CreateChild/CreateChildren.
    # Frozen afte __init__() call.
    self._private_children = NestedMap()
    # Child layers created by this layer. A well-formed layer should
    # have self._private_children equals to self._children_list. I.e.,
    # all child layers are created using CreateChild/CreateChildren.
    self._children_list = []
    # Keep track of vars/params that are being updated during the forward pass.
    # This is a temporary storage that gets cleared and updated for each FProp()
    # call.
    self._forward_updated_vars = py_utils.ThreadLocalDict()

    # Variable status.
    self._create_variables_status = CreateLayerVariablesStatus.NOT_ENABLED

    # Some sanity checks.
    # weight_split_dims_mapping and activation_split_dims_mapping specifies how
    # certain weights and activations of this layer and its children layers
    # should be sharded.
    assert isinstance(self._params.weight_split_dims_mapping, py_utils.Params)
    assert isinstance(self._params.activation_split_dims_mapping,
                      py_utils.Params)

  def PrepareFProp(self) -> None:
    """Prepares this layer for FProp()."""
    assert (
        self._create_variables_status == CreateLayerVariablesStatus.COMPLETED)
    tf.nest.map_structure(lambda x: x.PrepareFProp(), self._private_children)
    forward_updated_vars = tf.nest.map_structure(lambda v: None,
                                                 self._private_vars)
    self._forward_updated_vars.dict = forward_updated_vars

  def ForwardUpdateVar(self, name: str, new_val: JTensor) -> None:
    """Update var 'name' in the forward pass."""
    assert name in self._private_vars
    # TODO(yonghui): Maybe lift the constraint below.
    # A param can only be updated once.
    assert self._forward_updated_vars.dict[name] is None
    # Only non-trainable variables can be updated in the forward pass.
    assert VarNotTrainable(self.vars[name])
    self._forward_updated_vars.dict[name] = new_val

  def FProp(self, theta: Any, *args: Any, **kwargs: Any) -> Any:
    """Forward propagation.

    Note, this function is almost pure, except for the following elements:
    - prng_keys, global_step: they are fetched from a global thread-local
    context.
    - forward updated vars: they are temporarily stored in the layer local and
      thread-local forward_updated_vars dict.
    - summaries: they are stored in a global thread local dict.

    The central interface that subclasses should implement. The caller
    calls `FProp` with a `theta` dictionary. E.g.::

        foo = InstanceOfASubClassOfFoo(params)
        y = foo.FProp(foo.theta, x)

    The implementation of `FProp()` computes a function given
    the theta and the inputs. E.g.::

        subs = self.children
        inputs = args[0]
        a0 = subs.linear.FProp(theta.linear, inputs)
        a1 = subs.softmax.FProp(theta.softmax, a0)
        # The same layer applied twice.
        a2 = subs.linear.FProp(theta.linear, a1)
        return a2

    All the params needed by this layer and its sublayers are accessed through
    theta, with the exception of pseudo random number generator keys and a
    global step, which are accessed through global NextPrngKey() and
    CurGlobalStep().

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      *args: List args.
      **kwargs: Keyward args.
    """
    del theta
    del args
    del kwargs
    raise NotImplementedError('Abstract method of %s' % self)

  @property
  def params(self) -> BaseLayerParamsT:
    """Returns the params upon which this layer is built."""
    return self._params

  @property
  def jax_context(self) -> JaxContext:
    return CurJaxContext()

  @property
  def do_eval(self) -> bool:
    return self.jax_context.do_eval

  @property
  def parent(self) -> Optional[BaseLayerT]:
    """None if self is the root layer, otherwise the parent layer of self."""
    return self._parent

  @property
  def path(self) -> str:
    """Returns a '.'-separated string with all layer names from the root."""
    if self.parent:
      return self.parent.path + '.' + self.params.name
    else:
      return self.params.name

  @property
  def fprop_dtype(self) -> Any:
    if self.params.fprop_dtype is not None:
      return self.params.fprop_dtype
    else:
      return self.params.dtype

  @property
  def children(self) -> NestedMap:
    """Returns children layers of this layer in a `.NestedMap`."""
    return self._private_children

  def __getattr__(self, name: str) -> Any:
    """Returns the child layer of the given name."""
    if name == '_private_children':
      # Raising AttributeError without custom message triggers normal python
      # handling of __getattr__ AttributeError.
      raise AttributeError()
    if name in self._private_children:
      return self._private_children[name]
    elif (hasattr(type(self), name) and
          isinstance(getattr(type(self), name), property)):
      # There was an AttributeError raised by a property getter.
      # Call property getter again directly to raise the same error.
      return getattr(type(self), name).fget(self)
    else:
      raise AttributeError('%s is not a sub-layer of %s.' % (name, self))

  def GetDescendant(self, path: str) -> BaseLayerT:
    """Returns a descendant layer given the path.

    NOTE(yonghui): This GetDescendant is not complete. It is not able to descent
    into list/tuple substructures.

    Args:
      path: a comma separated string denoting a descendant of this layer.

    Returns:
      The descendant layer.

    Raises:
      KeyError: if the descendant is not found.
    """
    sub = self
    if path:
      for k in path.split('.'):
        if k not in sub.children:
          raise KeyError('%s not found in %s' % (k, list(sub.children.keys())))
        sub = sub.children[k]
    return sub

  @property
  def vars(self) -> NestedMap:
    """Returns variables of this layer and its children in a `.NestedMap`."""
    if self._create_variables_status != CreateLayerVariablesStatus.COMPLETED:
      raise ValueError(
          'Cannot access vars for layer %s before they have been created.' %
          self.params.cls)
    ret = self._private_children.Transform(lambda x: x.vars)
    for k in self._private_vars.keys():
      ret[k] = self._private_vars[k]
    return ret

  @property
  def total_num_vars(self) -> int:
    """Returns the total number of variables of this layer."""
    var_specs = py_utils.Flatten(self.vars)
    count = 0
    for v in var_specs:
      v_shape = list(v.shape)
      if not v_shape:
        v_shape = [1]
      if v.repeat_prefix:
        v_shape = list(v.repeat_prefix) + list(v_shape)
      count += np.prod(v_shape)
    return count

  @property
  def forward_updated_vars(self) -> NestedMap:
    """Returns variables updated during the last forward pass."""
    ret = self._private_children.Transform(lambda x: x.forward_updated_vars)
    for k in self._forward_updated_vars.dict.keys():
      ret[k] = self._forward_updated_vars.dict[k]
    return ret

  def _CheckName(self, name: str) -> None:
    """Asserts name's validity."""
    NestedMap.CheckKey(name)
    assert name not in self._private_vars, (
        '%s exists in vars, %s' % (name, list(self._private_vars.keys())))
    assert name not in self._private_children, (
        '%s exists in children, %s' %
        (name, list(self._private_children.keys())))

  def CreateVariable(self,
                     name: str,
                     var_params: ParamsT,
                     trainable: bool = True) -> None:
    """Create a variable of this layer according to the parameter `var_params`.

    E.g.::

        def CreateLayerVariables(self):
          self.CreateVariable(
              'weight', WeightParams(shape=[100, 100]))

    Args:
      name: Variable name which is used as the key into vars/theta.
      var_params: `Params` used to create the variable.
      trainable: whether or not this param is trainable.
    """
    var_params = var_params.Copy()
    p = self.params
    var_params.repeat_prefix = p.repeat_prefix
    var_params.repeat_prefix_split_dims_mapping = (
        p.repeat_prefix_split_dims_mapping)

    if p.device_mesh is not None:
      if (len([dim for dim in var_params.shape if dim > 1]) > 1 and
          var_params.tensor_split_dims_mapping is None):
        logging.warning('tensor_split_dims_mapping missing for %s.%s: shape=%s',
                        self.path, name, var_params.shape)
      if var_params.tensor_split_dims_mapping is not None:
        assert len(var_params.tensor_split_dims_mapping) == len(
            var_params.shape)
      if var_params.repeat_prefix_split_dims_mapping is not None:
        assert len(var_params.repeat_prefix) == len(
            var_params.repeat_prefix_split_dims_mapping)
    if self._create_variables_status == CreateLayerVariablesStatus.NOT_ENABLED:
      raise ValueError(
          'CreateVariable call is not enabled yet.'
          'CreateVariable should be called in CreateLayerVariables.')
    if self._create_variables_status == CreateLayerVariablesStatus.COMPLETED:
      raise ValueError(
          'CreateVariable call after variable creation has completed! '
          'CreateVariable should be called in CreateLayerVariables.')

    self._CheckName(name)

    if var_params.collections is None:
      var_params.collections = []

    if (p.skip_lp_regularization and
        SKIP_LP_REGULARIZATION not in var_params.collections):
      var_params.collections = var_params.collections + [SKIP_LP_REGULARIZATION]

    if (not trainable) and (NON_TRAINABLE not in var_params.collections):
      var_params.collections = var_params.collections + [NON_TRAINABLE]

    # Only keep a record of variables to be created for now.
    self._private_vars[name] = var_params

  def InstantiateVariableConfigs(self) -> None:
    """Instantiates variable configs for this layer and all its sub-layers.

    A variable config is a Params object containing all the meta information
    about a variable. Meta information included in the param includes: variable
    shape, how the variable should be initialized, whether or not the variable
    should be sharded, whether or not the variable should subject to certain
    regularization during training, whether or not the variable is learned
    through back-prop, etc. Unlike other Jax frameworks, Lingvo-Jax has the
    constraint that model variables are known apriori, before any training data
    is fed to the network.

    InstantiateVariableConfigs can only be called once. Configs for all
    variables for this layer and its sub-layers are instantiated during this
    function call.

    InstantiateVariableConfigs doesn't create the variables themselves.
    Variables are created during the InstantiateVariables() function call, and
    can be done 0 times, or multiple times.

    DO NOT OVERRIDE. Override self.CreateLayerVariables instead.
    """
    assert (
        self._create_variables_status == CreateLayerVariablesStatus.NOT_ENABLED)
    self._create_variables_status = CreateLayerVariablesStatus.ENABLED
    flattened_children = self._private_children.Flatten()
    for child in flattened_children:
      assert isinstance(child, BaseLayer)
      child.InstantiateVariableConfigs()
    self.CreateLayerVariables()
    self._create_variables_status = CreateLayerVariablesStatus.COMPLETED

  def InstantiateVariables(self, prng_key: PRNGKey) -> NestedMap:
    """Creates variables for this layer and its children layers.

    Note: this function can be called multple times. With the same prng_key as
    input, the output variables will be identical.

    Args:
      prng_key: A rank-2 tensor of prng keys.

    Returns:
      A NestedMap of Variables for this layer and its sub-layers.

    DO NOT OVERRIDE. Override self.CreateLayerVariables instead.
    """
    if self._create_variables_status != CreateLayerVariablesStatus.COMPLETED:
      self.InstantiateVariableConfigs()

    flattened_children = self._private_children.Flatten()
    children_vars = []
    prng_key, *subkeys = jax.random.split(prng_key, len(flattened_children) + 1)
    for child, subkey in zip(flattened_children, subkeys):
      assert isinstance(child, BaseLayer)
      children_vars.append(child.InstantiateVariables(subkey))
    children_vars_map = self._private_children.Pack(children_vars)
    assert isinstance(children_vars_map, NestedMap)

    self_path = self.path
    prng_key, *subkeys = jax.random.split(prng_key, len(self._private_vars) + 1)
    for (v_k, v_p), subkey in zip(self._private_vars.items(), subkeys):
      var_full_name = self_path + '.' + v_k
      children_vars_map[v_k] = InitVar(var_full_name, v_p, subkey)

    return children_vars_map

  def CreateLayerVariables(self) -> None:
    """Creates layer variables for this layer.

    Subclasses should override this function.

    This function merely records variables to be created. Variable are actually
    being created in the InstantiateVariables() function.
    """
    pass

  def CreateChild(self, name: str, params: BaseLayerParamsT) -> None:
    """Creates a sub layer.

    The created sub layer can be accessed by `name`. E.g.::

        self.CreateChild('foo', foo_params)
        self.foo.FProp...

    or:

        self.children['foo'].Fprop...
        self.children.foo.Fprop...

    If the layer does not have a name set, i.e. foo_params.name is None, then
    its name will be set to `name`.

    Args:
      name: Sub layer name which is used as the key into vars/theta.
      params: `Hyperparams` object to instantiate a layer.
    """
    if hasattr(self, '_disable_create_child') and self._disable_create_child:
      raise ValueError('Attempting to call CreateChild outside of __init__.')
    self._CheckName(name)
    p = self.CopyBaseParams(self.params, params.Copy())
    if not p.name:
      p.name = name
    child = p.Instantiate()
    self._private_children[name] = child

  def CreateChildren(
      self, name: str, params: Union[Sequence[BaseLayerParamsT],
                                     Mapping[str, BaseLayerParamsT]]
  ) -> None:
    """Creates a list or dict of sub layers.

    The created sub layer list can be accessed by `name`. E.g.::

        self.CreateChildren('foo', ...)
        self.foo[10].FProp...

    or::

        self.children['foo'][10].Fprop...
        self.children.foo[10].Fprop...

    Args:
      name: The name for the sub layers, which is used as the key into
        vars/theta.
      params: a list or dict of `Hyperparams` objects to create.
    """
    if hasattr(self, '_disable_create_child') and self._disable_create_child:
      raise ValueError('Attempting to call CreateChildren outside of __init__.')
    self._CheckName(name)
    params = NestedMap.FromNestedDict(params)

    uid = itertools.count()

    def Instantiate(p: InstantiableParams) -> BaseLayerT:
      p = self.CopyBaseParams(self.params, p.Copy())
      if not p.name:
        p.name = '%s_%d' % (name, next(uid))
      return p.Instantiate()

    self._private_children[name] = NestedMap(
        sub=params).Transform(Instantiate).sub

  def _AutoAddChild(self, child: BaseLayerT) -> None:
    """Records that a layer `child` is instantiated by this layer.

    This method should only be called internally by BaseLayerMeta.

    Args:
      child: A sub-layer of this layer.
    """
    self._children_list.append(child)

  def _VerifyChildren(self) -> None:
    """Verify all children created by this layer are via `CreateChild(ren)`."""
    created_children = self._private_children.Flatten()
    for v in self._children_list:
      if v not in created_children:
        logging.info([
            (child.params.name, type(child)) for child in created_children
        ])
        raise ValueError(
            '%s is not created by BaseLayer.CreateChild(ren) in %r.' %
            (v.params.name, self))

  def _CastToFPropDtype(self, value: Any) -> Any:
    """Casts values to the desired dtype."""

    def _Cast(x):
      if x is None:
        return None
      if self.fprop_dtype != x.dtype:
        if jnp.issubdtype(x.dtype, jnp.floating):
          return x.astype(self.fprop_dtype)
      return x

    return tf.nest.map_structure(_Cast, value)


def AssertHasShape(t: JTensor, shape: Sequence[int]) -> None:
  assert t.ndim == len(shape)
  for i in range(t.ndim):
    assert t.shape[i] == shape[i] or shape[i] == -1
