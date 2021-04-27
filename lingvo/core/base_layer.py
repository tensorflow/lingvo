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
"""Base class for all layers."""

import abc
import collections
import contextlib
import copy
import enum
import itertools
import re
from typing import Callable, List, Mapping, Optional, Type, TypeVar, Union
import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import gshard_utils
from lingvo.core import hyperparams
from lingvo.core import py_utils

FLAGS = tf.flags.FLAGS


_LAYER_STACK = py_utils.ThreadLocalStack()
_CREATE_VARIABLES_STACK = py_utils.ThreadLocalStack()


class Accumulator:
  """Layers can register accumulators to persist step-level state.

  Accumulators must be represented by a Tensor of a fixed shape. The default
  value must be supplied by overriding DefaultValue(). It is important that
  the default tensor value is created on each call in order to avoid
  accumulators leaking to different graphs.

  Accumulators can be enabled (default) or disabled by pairing
  Disable()/Enable() calls. When disabled, the accumulator will only return
  the default value and will silently drop calls to SetValue(). When computing
  gradients that may touch accumulators, calls should be bracketed with
  Disable()/Enable().

  Care must be taken when manipulating accumulators across Defun boundaries.
  Typically, values for all accumulators in a layer must be explicitly
  retrieved and passed in to the Defun scope by calling
  layer.GetAccumulatorValues(), marshalling into the Defun and setting them
  via layer.SetAccumulatorValues(). The reverse must be done on return.
  """

  def __init__(self):
    # None for initial value or the current Tensor value.
    self._value = None
    self._disable_count = 0

  @property
  def is_disabled(self):
    """Whether the accumulator is disabled."""
    return self._disable_count > 0

  def Disable(self):
    """Disables the accumulator (must be balanced with Enable)."""
    self._disable_count += 1

  def Enable(self):
    """Enables the accumulator (must balance a Disable)."""
    assert self._disable_count > 0, 'Unbalanced Accumulator Enable/Disable'
    self._disable_count -= 1

  def GetValue(self):
    """Gets the current value of the accumulator Tensor."""
    if self.is_disabled or self._value is None:
      return self.DefaultValue()
    else:
      return self._value

  def SetValue(self, value):
    """Sets the current value of the accumulator Tensor."""
    if not self.is_disabled:
      self._value = value

  def Reset(self):
    """Resets the accumulator to its default value."""
    if not self.is_disabled:
      self._value = None

  def DefaultValue(self):
    raise NotImplementedError('DefaultValue must be implemented')


def _BaseLayerInitWrapper(func):  # pylint: disable=invalid-name
  """A decorator for layer's __init__.

  Args:
    func: The __init__ method of `BaseLayer`'s subclasses.

  Returns:
    A decorator wrapper for layer's initializer. Note that this wrapper can
    be called multiple times for the same layer instance, once for each
    __init__() for classes on the class hierarchy.
  """

  def Wrapper(self, *args, **kwargs):
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

    if not stack:
      # Outermost layer just finished __init__.
      if self.cluster.immediately_instantiate_variables:
        self.InstantiateVariables()

  return Wrapper


def RecursiveFindLayerParams(params):
  """Returns all params that define a layer."""
  if not isinstance(params, hyperparams.Params):
    return []
  layer_params = []
  if hasattr(params, 'cls') and issubclass(params.cls, BaseLayer):
    layer_params.append(params)
  for _, p in params.IterParams():
    if isinstance(p, (list, tuple)):
      for item in p:
        layer_params.extend(RecursiveFindLayerParams(item))
    elif isinstance(p, dict):
      for item in p.items():
        layer_params.extend(RecursiveFindLayerParams(item))
    else:
      layer_params.extend(RecursiveFindLayerParams(p))
  return layer_params


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


class ABCLayerMeta(BaseLayerMeta, abc.ABCMeta):
  pass


# NamedTuple that records the metadata for creating a variable.
# For internal use only. Subclasses of BaseLayer should use
# self.CreateVariable() to create variables.
CreateVariableMeta = collections.namedtuple(
    'CreateVariableMeta', ['var_params', 'theta_fn', 'kwargs'])


class _CreateLayerVariablesStatus(enum.Enum):
  NOT_CALLED = 1
  IN_PROGRESS = 2
  COMPLETED = 3
  PER_SPLIT_COMPLETED = 4


LAYER_WT = 'layer_weight_variable'


BaseLayerS = TypeVar('BaseLayerS', bound='BaseLayer')
BaseLayerT = TypeVar('BaseLayerT', bound='BaseLayer')
BaseLayerParamsT = hyperparams.InstantiableParams[BaseLayerT]


class BaseLayer(tf.Module, metaclass=BaseLayerMeta):
  r"""Base class for all the layer object.

  As this BaseLayer is a proper sub-class of tf.Module, it supports proper
  tracking and reflection of key constituents such as variables and submodules.

  self.submodules returns a list of submodules that are reachable through
  recursive member access from self.

  self.variables returns a list of Variables that are reachable through
  recursive member access from self.

  self(\*args, \*\*kwargs) carries out computation on the input args and kwargs.
  """

  # Set to an inference driver name if this is an inference specialization
  # class.
  _INFERENCE_DRIVER_NAME = None

  @classmethod
  def Params(cls: Type[BaseLayerT]) -> BaseLayerParamsT:
    """Returns the layer params."""
    p = hyperparams.InstantiableParams(cls)
    p.Define('inference_driver_name', cls._INFERENCE_DRIVER_NAME,
             'Name of the inference driver used to construct this layer.')
    p.Define('name', '', 'Name of this layer object.')
    p.Define('dtype', tf.float32, 'Datatype to use.')
    # None value will make FProp use dtype instead of fprop_dtype.
    # TODO(lepikhin): all @tf.Defun should use p.fprop_dtype if it is set.
    p.Define('fprop_dtype', None, 'Activations datatype to use.')
    p.Define(
        'random_seed', None, 'Random seed for deterministic unittests. This '
        'is inherited by child layers if they do not set a random_seed.')
    p.Define('vn', py_utils.DefaultVN(),
             'How variational noise should be applied.')
    p.Define(
        'params_init', py_utils.DefaultParamInit(),
        'How model weights should be initialized. Not to be confused with '
        'hyperparams.')
    # Makes additional alterations for graphs being used for inference.
    p.Define('is_inference', None, 'True if in inference mode.')
    # In addition to is_inference, indicate that the inference graph is
    # for a single step.
    p.Define(
        'allow_implicit_capture', None,
        'When using Defuns, code often asserts that the Defun does not '
        'capture undeclared inputs. This eliminates a source of bugs '
        'at the expense of making some kinds of models or utilities '
        'hard/impossible to use. Setting this to True/False (versus None) '
        'causes the setting to apply to this layer and its children.')
    p.Define(
        'skip_lp_regularization', None,
        'If True, all variables in this layer will skip Lp regularization. '
        'If None/False, only variables explicitly in the '
        'SKIP_LP_REGULARIZATION collection will skip Lp regularization. '
        'Also propagated to child layers with default settings (None).')
    # SPMD partition related params.
    p.Define(
        'device_mesh', None,
        'A numpy.ndarray specifying the topology of a device mesh to place the'
        ' computations onto. If device_mesh is None, it is assumed to be a'
        ' single device. Here are some examples:'
        ' np.array([0, 1, 2, 3, 4, 5, 6, 7]) which is a 1d mesh with 8 devices,'
        ' np.array([[0, 1, 2, 3], [4, 5, 6, 7]]) which is 2d matrix of 8'
        ' devices.')
    p.Define(
        'weight_split_dims_mapping', None,
        'Relevant only if device_mesh above is not None. If not None, it '
        'specifies how weight of this layer or those of the sublayers should '
        'be sharded over device mesh. ')
    p.Define(
        'activation_split_dims_mapping', None,
        'Relevant only if device_mesh above is not None. If not None, it '
        'specifies how activation of this layer or those of the sublayers '
        'should be sharded over device mesh. ')
    return p

  @staticmethod
  def CopyBaseParams(from_params: hyperparams.InstantiableParams[BaseLayerS],
                     to_params: BaseLayerParamsT) -> BaseLayerParamsT:
    """Copies BaseLayer params from `from_params` to `to_params`."""
    assert issubclass(from_params.cls, BaseLayer)
    assert issubclass(to_params.cls, BaseLayer)
    # Copy-over the BaseLayer params.
    if to_params.dtype == tf.float32:
      to_params.dtype = from_params.dtype
    if to_params.fprop_dtype is None:
      to_params.fprop_dtype = from_params.fprop_dtype
    if to_params.random_seed is None:
      to_params.random_seed = from_params.random_seed
    if to_params.is_inference is None:
      to_params.is_inference = from_params.is_inference
    if to_params.allow_implicit_capture is None:
      to_params.allow_implicit_capture = from_params.allow_implicit_capture
    if to_params.skip_lp_regularization is None:
      to_params.skip_lp_regularization = from_params.skip_lp_regularization

    if to_params.device_mesh is None:
      to_params.device_mesh = copy.deepcopy(from_params.device_mesh)

    # Only copy from base when vn config is using the default setting.
    if to_params.vn == py_utils.DefaultVN():
      to_params.vn = from_params.vn.Copy()

    # TODO(rpang): derive to_params.params_init.seed from
    # from_params.params_init.seed if it is specified in 'from_params' and not
    # in 'to_params'.
    if py_utils.IsDefaultParamInit(to_params.params_init):
      # Copy over params_init as well.
      to_params.params_init = from_params.params_init.Copy()
    return to_params

  def __init__(self: BaseLayerT, params: BaseLayerParamsT) -> None:
    """Layer constructor.

    Args:
      params: A params used to construct this layer.
    """
    assert params.name, (
        'Layer params for %s must have a "name"' % self.__class__.__name__)

    tf_module_name = params.name
    tf_module_name = re.sub('[^a-zA-Z0-9_]+', '_', tf_module_name)
    tf_module_name = 'bbf_' + self.__class__.__name__ + '_' + tf_module_name
    py_utils.NestedMap.CheckKey(tf_module_name)

    # initialize the base class.
    super().__init__(tf_module_name)

    # Note AutoTracking doesn't work properly due to its inability to walk
    # through py_utils.NestedMap data structures which are used widely
    # throughout the Lingvo codebase. Also there seems to be some performance
    # hit in turning on auto-tracking in constructing graphs. For now, we
    # disable auto-tracking.
    # TODO(lingvo): Re-enable auto-tracking when fuller support is
    # added for key data structures used in Lingvo, and performance issue is
    # debugged more and understood better.
    self._setattr_tracking = False

    self._parent = None
    for parent in reversed(_LAYER_STACK.stack):
      if parent is not self:
        self._parent = parent
        break
    self._params = params.Copy()
    tf.logging.debug('Creating layer %s with params: \n %s \n',
                     self.__class__.__name__, str(params))
    # Vars created by this layer.
    self._private_vars = py_utils.NestedMap()
    # Theta derived from this layer's vars.
    self._private_theta = py_utils.NestedMap()
    # A simple transformation before used by the forward computation. Its
    # signature must be (tf.Tensor) -> (tf.Tensor).
    self._private_theta_fn = py_utils.NestedMap()
    # Child layers created by this layer through CreateChild/CreateChildren.
    self._private_children = py_utils.NestedMap()
    # Child layers created by this layer. A well-formed layer should
    # have self._private_children equals to self._children_list. I.e.,
    # all child layers are created using CreateChild/CreateChildren.
    self._children_list = []
    # Extra theta's not directly correspond to any underlying vars. For example,
    # the concatenated sharded variables.
    self._extra_theta = py_utils.NestedMap()
    # All registered accumulators.
    self._private_accumulators = py_utils.NestedMap()
    # Layer-private functions. Add with AddFunction.
    self._private_fns = dict()
    # Mapping from variable names to its symbolic shape.
    # self._var_symbolic_shape_map['var_name'] will be a tuple of integers or
    # symbolic expressions, one for each dimension of the variable.
    self._var_symbolic_shape_map = {}

    self._is_variable_free = False
    self._variables_to_create = {}
    self._create_variables_status = _CreateLayerVariablesStatus.NOT_CALLED
    # Keep track of the tf.variable_scope(p.name) this layer creates so we can
    # reenter it without creating a new one.
    self._self_variable_scope = None

  def SetVariableFree(self, value: bool = True) -> None:
    """Marks this layer as having no variables.

    Note that this status affects sublayers and child layers too.

    Args:
      value: True to set layer as variable free.
    """
    if self._create_variables_status != _CreateLayerVariablesStatus.NOT_CALLED:
      raise ValueError(
          'Variable free status for %s must be set before InstantiateVariables().'
          % self.params.cls)
    if self._variables_to_create:
      raise ValueError('Cannot set layer %s with variables as variable free.' %
                       self.params.cls)
    self._is_variable_free = value

  def FPropDefaultTheta(self, *args, **kwargs):
    """Calls `FProp`."""
    return self.FProp(self.theta, *args, **kwargs)

  def __call__(self, *args, **kwargs):
    """Forwards call to FPropDefaultTheta."""
    return self.FPropDefaultTheta(*args, **kwargs)

  def FProp(self, theta, *args, **kwargs):
    """Forward propagation.

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

    Args:
      theta: A `.NestedMap` object containing weights' values of this
        layer and its children layers.
      *args: List args.
      **kwargs: Keyward args.
    """
    del theta
    del args
    del kwargs
    raise NotImplementedError('Abstract method of %s' % self)

  @classmethod
  def FPropMeta(cls, params, *args, **kwargs):
    """Returns metadata about the `FProp` computation for this layer.

    **Experimental feature.**
    Don't use or depend on it without consulting Lingvo authors.

    E.g.::

        p = SomeComplexLayer.Params()
        meta = p.cls.FPropMeta(p, tshape.Shape([128, 20, 50, 'channels']))

    `meta.flops` gives an estimate count of floating point operations done by
    one `FProp` given an input tensor of shape [128, 20, 50, channels].
    `meta.out_shapes` is a tuple of TShape, which tells you what shape
    of tensors this layer will return.

    Args:
      params: The param of a layer of this layer type.
      *args: Corresponds to FProp with Tensors replaced by `TensorShape`.
      **kwargs: Corresponds to FProp with Tensors replaced by `TensorShape`.

    Returns:
      A `.NestedMap` with

      - flops - The estimated number of floating point operations incurred by
        this fprop.
      - out_shapes - A tuple of `TShape`. I.e., `out_shapes[i]`
        represents the shape of the `i`-th returned tensor of the fprop.
    """
    raise NotImplementedError('FPropMeta of %s' % cls)

  @property
  def params(self) -> BaseLayerParamsT:
    """Returns the params upon which this layer is built."""
    return self._params

  @property
  def cluster(self):
    """Returns the current cluster configuration."""
    return cluster_factory.Current()

  @property
  def do_eval(self) -> bool:
    return self.cluster.do_eval

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
  def layer_type(self) -> str:
    """Returns layer type prefixed with 'lingvo.'."""
    return 'lingvo.' + self.__class__.__name__

  @property
  def children(self) -> py_utils.NestedMap:
    """Returns children layers of this layer in a `.NestedMap`."""
    return self._private_children

  def __getattr__(self, name: str):
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
  def vars(self):
    """Returns variables of this layer and its children in a `.NestedMap`."""
    if self._is_variable_free:
      return self._private_children.Transform(lambda _: py_utils.NestedMap())
    if self._create_variables_status == _CreateLayerVariablesStatus.NOT_CALLED:
      raise ValueError(
          'Cannot access vars for layer %s before they have been created.' %
          self.params.cls)
    ret = self._private_children.Transform(lambda x: x.vars)
    for k in self._private_vars.keys():
      ret[k] = self._private_vars[k]
    return ret

  @property
  def theta(self):
    """Returns theta of this layer and its children in a `.NestedMap`."""
    if self._is_variable_free:
      return self._private_children.Transform(lambda _: py_utils.NestedMap())
    if self._create_variables_status == _CreateLayerVariablesStatus.NOT_CALLED:
      raise ValueError(
          'Cannot access theta for layer %s before they have been created.' %
          self.params.cls)
    ret = self._private_children.Transform(lambda x: x.theta)

    private_theta = self._private_theta.DeepCopy()
    for name, theta_fn in self._private_theta_fn.FlattenItems():
      private_theta[name] = theta_fn(private_theta[name])

    if (self._params.fprop_dtype is not None and
        self._params.fprop_dtype != self._params.dtype):

      def MaybeCastToFPropDtype(x):
        # Need to check `.base_dtype` as x.dtype may be tf.float32_ref.
        if x is not None and x.dtype.base_dtype == self._params.dtype:
          return tf.cast(x, self._params.fprop_dtype)
        else:
          return x

      private_theta = private_theta.Transform(MaybeCastToFPropDtype)

    ret.update(private_theta)
    return ret

  @property
  def accumulators(self):
    """Returns `.NestedMap` of `Accumulator` instances for this and children."""
    ret = self._private_children.Transform(lambda x: x.accumulators)
    for k, acc in self._private_accumulators.items():
      ret[k] = acc
    return ret

  @property
  def fns(self):
    """Returns a read-only view of layer local functions.

    Functions can be accessed by index (['name']) or attribute notation
    (`fns.foo`).

    Returns:
      Read-only attribute accessible dict view of the layer's function library.
    """
    return py_utils.ReadOnlyAttrDictView(self._private_fns)

  def AddFunction(self, name, f, replace=False):
    """Adds a function to the layer's `fns` collection.

    This should be used to add op-like functions specific to the operation
    of the layer and its children. Such functions should be added in `__init__`
    and may either be raw python functions or TensorFlow Defuns. This
    facility is just a mechanism for organizing them and having basic checks
    on name collisions.

    Args:
      name: The function name. It will be accessible as `self.fns.{name}`.
      f: The function body.
      replace: Whether to replace an existing function (default False).

    Raises:
      AttributeError: If the function already exists and replace == False.
    """
    py_utils.NestedMap.CheckKey(name)
    if not replace:
      if name in self._private_fns:
        raise AttributeError(
            'Function "%s" is already defined on layer "%r"' % (name, self))
      self._private_fns[name] = f

  def _CheckName(self, name: str) -> None:
    """Asserts name's validity."""
    py_utils.NestedMap.CheckKey(name)
    assert name not in self._private_vars, (
        '%s exists in vars, %s' % (name, list(self._private_vars.keys())))
    assert name not in self._private_theta, (
        '%s exists in theta, %s' % (name, list(self._private_theta.keys())))
    assert name not in self._private_children, ('%s exists in children, %s' % (
        name, list(self._private_children.keys())))
    assert name not in self._private_accumulators, (
        '%s exists in global_accumulator: %s' %
        (name, list(self._private_accumulators.keys())))

  def _VariableCollections(self) -> List[str]:
    return [LAYER_WT, '%s_vars' % self.__class__.__name__]

  def RegisterAccumulator(self, name, acc):
    """Registers an accumulator for this layer.

    An accumulator is used to propagate some state to a future point,
    where it is acted on (typically as part of `PostTrainingStepUpdate`). This
    mechanism allows for arbitrarily nested parts of a model to export state
    back to the global scope. Accumulators must be specially handled
    when crossing into `Defun` or recurrent scopes. By abstracting the
    mechanism, it allows all such state to be handled uniformly and generically.

    Example (typically from `__init__`)::

        class MyAccumulator(base_layer.Accumulator):
          def DefaultValue(self):
            # [count, min, max]
            return tf.convert_to_tensor([0.0, 0.0, 0.0])
          def Update(self, state1):
            state0 = self.GetValue()
            self.SetValue(tf.stack([
                state0[0] + state1[0],
                tf.minimum(state0[1], state1[1]),
                tf.maximum(state0[2], state1[2])]))

        self.RegisterAccumulator('mytracker', acc)

    Later, access the current value and update it::

        acc = self.accumulators.mytracker
        acc.Update(tf.convert_to_tensor([1.0, batch_min, batch_max]))

    Then, typically in `PostTrainingStepUpdate`::

        acc = self.accumulator.mytracker
        acc_value = acc.GetValue()
        # Do something with the value.
        acc.Reset()

    Args:
      name: The accumulator name. Shares a namespace with children, vars and
          extra theta.
      acc: An `Accumulator` instance.
    """
    self._CheckName(name)
    self._private_accumulators[name] = acc

  def GetAccumulatorValues(self):
    """Recursively gets values of all accumulators.

    Returns:
      `.NestedMap` of Tensors for each registered accumulator.
    """
    return self.accumulators.Transform(lambda acc: acc.GetValue())

  def SetAccumulatorValues(self, new_values_nmap):
    """Recursively sets the values of all accumulators from a map.

    Args:
      new_values_nmap: `.NestedMap` of accumulator name:Tensor.
    """
    accumulator_list = self.accumulators.Flatten()
    value_list = new_values_nmap.Flatten()
    for acc, value in zip(accumulator_list, value_list):
      acc.SetValue(value)

  def GetVariableSymbolicShape(self, var_name):
    """Returns the variable's symbolic shape."""
    return self._var_symbolic_shape_map.get(var_name, None)

  def CreateVariable(self,
                     name: str,
                     var_params: py_utils.WeightParams,
                     theta_fn: Optional[Callable[[tf.Tensor],
                                                 tf.Tensor]] = None,
                     **kwargs) -> None:
    """Create a variable of this layer according to the parameter `var_params`.

    E.g.::

        def __init__(self, ...):    # A layer's constructor
          self.CreateVariable(
              'weight', py_utils.WeightParams(shape=[100, 100]))

    `theta_fn` is used to apply a simple transformation on the created
    variable's value before used by the forward computation. E.g., to
    add the global variational noise according to this layer's
    parameter, one can do::

        def __init__(self, ...):    # A layer's constructor
          self.CreateVariable(
            name='weight',
            var_params=py_utils.WeightParams(shape=[100, 100]),
            theta_fn=self.AddVN)

    In some contexts, eg. TPU training, variables may not be created immediately
    but rather the creation request will be cached and created later via a call
    to layer.InstantiateVariables().

    Args:
      name: Variable name which is used as the key into vars/theta.
      var_params: `Params` used to create the variable.
      theta_fn: A python function that takes a variable's value and returns a
        new value to be used later for computation. Its signature must be
        (tf.Tensor) -> (tf.Tensor).
      **kwargs: Keyword args passed to `.py_utils.CreateVariable`.
    """
    if self.params.device_mesh is not None:
      if (len([dim for dim in var_params.shape if dim > 1]) > 1 and
          var_params.tensor_split_dims_mapping is None):
        tf.logging.warning(
            'tensor_split_dims_mapping missing for %s.%s: shape=%s', self.path,
            name, var_params.shape)
    if self._is_variable_free:
      raise ValueError('Cannot create variable in variable free layer.')
    if self._create_variables_status == _CreateLayerVariablesStatus.COMPLETED:
      raise ValueError(
          'CreateVariable call after variable creation has completed! '
          'CreateVariable should be called in __init__ or _CreateLayerVariables.'
      )
    self._CheckName(name)
    if (self.params.skip_lp_regularization and
        py_utils.SKIP_LP_REGULARIZATION not in var_params.collections):
      var_params = py_utils.WeightParams(
          shape=var_params.shape,
          dtype=var_params.dtype,
          init=var_params.init,
          collections=(var_params.collections +
                       [py_utils.SKIP_LP_REGULARIZATION]))
    self._var_symbolic_shape_map[name] = var_params.shape
    meta = CreateVariableMeta(
        var_params=var_params.Copy(),
        theta_fn=theta_fn,
        kwargs=kwargs)
    if self._create_variables_status == _CreateLayerVariablesStatus.IN_PROGRESS:
      # If InstantiateVariables has been called, create variable immediately.
      self._CreateVariableInternal(name, meta)
    else:
      # Otherwise cache the variable to be created.
      self._variables_to_create[name] = meta

  def _CreateVariableInternal(self, name: str,
                              meta: CreateVariableMeta) -> None:
    """Immediately creates the variable described by `meta`.

    DO NOT OVERRIDE. For internal use only. Subclasses of BaseLayer should use
    self.CreateVariable() to create variables.

    Args:
      name: The variable name.
      meta: A CreateVariableMeta describing the variable to be created.
    """
    meta.kwargs.setdefault('default_seed', self.params.random_seed)
    var = py_utils.CreateVariable(name, meta.var_params, **meta.kwargs)
    self._private_vars[name] = var
    if self.cluster.params.worker.gpus_per_replica > 0:
      # On GPU (which always trains a single step per session.run()), reference
      # a tensor in FProp to cache it on device and avoid extraneous sends from
      # reading variables from ps multiple times.
      with tf.device(var.device):
        value = tf.identity(var)
    else:
      # Pass the resource variable directly into the training loop.
      value = var

    # Due to b/174956514, we have to annotate the use of the variable once,
    # otherwise, the sharding annotation on the var will be ignored.
    # TODO(yonghui): Get rid of this once b/174956514 is fixed.
    if (meta.var_params.device_mesh is not None and
        var.shape.rank == len(meta.var_params.tensor_split_dims_mapping)):
      value = gshard_utils.MeshSplit(
          value,
          meta.var_params.device_mesh,
          meta.var_params.tensor_split_dims_mapping,
          use_sharding_op=True)

    if meta.theta_fn is not None:
      self._private_theta_fn[name] = meta.theta_fn

    self._private_theta[name] = value

  @contextlib.contextmanager
  def _SelfVariableScope(self):
    """Internal. Used to ensure the same variable & name scopes are used."""
    if not self._self_variable_scope:
      with tf.variable_scope(py_utils.SanitizeScopeKey(
          self.params.name)) as scope:
        self._self_variable_scope = scope
    with contextlib.ExitStack() as stack:
      stack.enter_context(
          tf.variable_scope(
              self._self_variable_scope, auxiliary_name_scope=False))
      stack.enter_context(
          tf.name_scope(self._self_variable_scope.original_name_scope))
      yield stack

  def InstantiateVariables(self) -> None:
    """Create variables for this layer and child layers.

    DO NOT OVERRIDE. Override self._CreateLayerVariables instead.
    """
    if self._create_variables_status != _CreateLayerVariablesStatus.NOT_CALLED:
      return
    self._create_variables_status = _CreateLayerVariablesStatus.IN_PROGRESS

    stack_size = len(_CREATE_VARIABLES_STACK.stack)
    _CREATE_VARIABLES_STACK.stack.append(self)
    try:
      self._CreateChildrenVariables()

      if not self._is_variable_free:
        with self._SelfVariableScope():
          for name, meta in list(self._variables_to_create.items()):
            self._CreateVariableInternal(name, meta)
          self._CreateLayerVariables()
    finally:
      assert _CREATE_VARIABLES_STACK.stack[-1] is self
      _CREATE_VARIABLES_STACK.stack.pop()
      assert len(_CREATE_VARIABLES_STACK.stack) == stack_size

    self._create_variables_status = _CreateLayerVariablesStatus.COMPLETED

    if not _CREATE_VARIABLES_STACK.stack:
      # Outermost layer just finished InstantiateVariables.
      self._VerifyVarsAndTheta()

  def _CreateChildrenVariables(self) -> None:
    """Create variables for child layers.

    Should be rarely overridden, only in cases when control over the context of
    children InstantiateVariables calls are needed. eg, if children variables
    need to be created inside of a specific context manager.

    There are a few cases of this in the codebase marked as for backwards
    compability. This is only to ensure that variable scopes remain compatible
    through the code migration. New layers should not copy that pattern, and
    instead follow the standard pattern of self.CreateChild() in __init__() and
    self.CreateVariable() in _CreateLayerVariables(). If you are okay with
    breaking old checkpoints, you can go ahead and delete those functions.
    """
    with self._SelfVariableScope():
      for child in self._children_list:
        if self._is_variable_free and not child._is_variable_free:  # pylint: disable=protected-access
          raise ValueError(
              'Variable free layer %s(%s) child %s(%s) has variables.' %
              (self.params.name, self.params.cls, child.params.name,
               child.params.cls))
        child.InstantiateVariables()

  def _CreateLayerVariables(self) -> None:
    """Actually create variables for this layer.

    Subclasses should override this function.

    Variables are created inside of tf.variable_scope(self.params.name).
    """
    pass

  def AddExtraTheta(self, theta_name: str, theta_value) -> None:
    """Add extra `theta` that doesn't directly correspond to `vars`."""
    self._CheckName(theta_name)
    self._private_theta[theta_name] = theta_value
    self._extra_theta[theta_name] = theta_value

  def AddVN(self, value):
    return py_utils.AddVN(self.params, value)

  def CreateChild(self, name: str, params: BaseLayerParamsT) -> None:
    """Create a sub layer.

    The created sub layer can be accessed by `name`. E.g.::

        self.CreateChild('foo', foo_params)
        self.foo.FProp...

    or::

        self.children['foo'].Fprop...
        self.children.foo.Fprop...

    If the layer does not have a name set, i.e. foo_params.name is None, then
    its name will be set to `name`. The layer's name is used as a variable_scope
    for its variables.

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
      self, name: str, params: Union[List[BaseLayerParamsT],
                                     Mapping[str, BaseLayerParamsT]]
  ) -> None:
    """Create a list or dict of sub layers.

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

    uid = itertools.count()

    def Instantiate(p):
      p = self.CopyBaseParams(self.params, p.Copy())
      if not p.name:
        p.name = '%s_%d' % (name, next(uid))
      return p.Instantiate()

    self._private_children[name] = py_utils.NestedMap(
        sub=params).Transform(Instantiate).sub

  def AddChild(self, name: str, children: BaseLayerT) -> None:
    """Add existing layer or layers as sublayer."""
    for child in py_utils.Flatten(children):
      assert isinstance(child, BaseLayer)
    self._CheckName(name)
    self._private_children[name] = children

  def _AutoAddChild(self, child: BaseLayerT) -> None:
    """Record that a layer `child` is instantiated by this layer.

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
        tf.logging.info([
            (child.params.name, type(child)) for child in created_children
        ])
        raise ValueError(
            '%s is not created by BaseLayer.CreateChild(ren) in %r.' %
            (v.params.name, self))

  def _VerifyVarsAndTheta(self) -> None:
    """Verify that vars and theta have the same nested structure."""
    for child in self._children_list:
      child._VerifyVarsAndTheta()  # pylint: disable=protected-access

    def MatchKeys(x, y):
      assert len(x) <= len(y)
      for k in x.keys():
        assert k in y, '%s not in %s.' % (k, y)
        if isinstance(x[k], py_utils.NestedMap):
          assert isinstance(y[k], py_utils.NestedMap), '%s is not a map' % y[k]
          MatchKeys(x[k], y[k])

    # NOTE: this check can be quadratically expensive. Maybe only
    # enable this in unittests.
    MatchKeys(self.vars, self.theta)

    # Make sure whatever not in self.vars are in self._extra_theta
    for k in self.theta.keys():
      assert k in self.vars or k in self._extra_theta

  def PostTrainingStepUpdate(self):
    """Returns a TF op which will be invoked at each training step.

    Subclasses of `BaseLayer` can implement this method. The method should
    return a TF op to be invoked during training after gradients are applied.
    """
    update_ops = [
        child.PostTrainingStepUpdate()
        for child in self._private_children.Flatten()
    ]
    return tf.group(*update_ops)

  def _CastToFPropDtype(self, value):

    def _Cast(x):
      if x is None:
        return None
      x = tf.convert_to_tensor(x)
      if not x.dtype.is_floating:
        return x
      return tf.cast(x, py_utils.FPropDtype(self.params))

    return tf.nest.map_structure(_Cast, value)


def IsLayerParams(x):
  return (isinstance(x, hyperparams.InstantiableParams) and
          issubclass(x.cls, BaseLayer))
