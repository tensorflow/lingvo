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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import six
import tensorflow as tf

from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import py_utils


class _LocalLayerStack(threading.local):

  def __init__(self):
    super(_LocalLayerStack, self).__init__()
    self.layer_stack = []


_LAYER_STACK = _LocalLayerStack()


class Accumulator(object):
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


def initializer(func):  # pylint: disable=invalid-name
  """A decorator for layer's __init__.

  Args:
    func: The __init__ method of `BaseLayer`'s subclasses.

  Returns:
    A decorator wrapper for layer's initializer.
  """

  def wrapper(self, *args, **kwargs):  # pylint: disable=invalid-name
    # Push back self (the current layer) to the stack.
    stack = _LAYER_STACK.layer_stack
    stack.append(self)
    try:
      # Calls the layer's real __init__ method.
      func(self, *args, **kwargs)
      # pylint: disable=protected-access
      self._CheckInvariants()
      assert id(stack[-1]) == id(self)
      if len(stack) > 1 and id(stack[-2]) != id(self):
        # Records the fact stack[-1] just created a sub-layer self.
        stack[-2]._AutoAddChild(self)
    finally:
      # Pop out self (the current layer).
      stack.pop()

  return wrapper


def DefaultVN():
  return py_utils.VariationalNoiseParams(None, False, False)


def RecursiveFindLayerParams(params):
  """Returns all params that define a layer."""
  layer_params = []
  if hasattr(params, 'cls') and issubclass(params.cls, BaseLayer):
    layer_params.append(params)
  for _, p in params.IterParams():
    if isinstance(p, hyperparams.Params):
      layer_params.extend(RecursiveFindLayerParams(p))
  return layer_params


LAYER_WT = 'layer_weight_variable'


class BaseLayer(object):
  """Base class for all the layer object."""

  # Set to an inference driver name if this is an inference specialization
  # class.
  _INFERENCE_DRIVER_NAME = None

  @classmethod
  def Params(cls):
    """Returns the layer params."""
    p = hyperparams.Params()
    p.Define('cls', cls, 'Cls that this param object is associated with.')
    p.Define('inference_driver_name', cls._INFERENCE_DRIVER_NAME,
             'Name of the inference driver used to construct this layer.')
    p.Define('name', '', 'Name of this layer object.')
    p.Define('dtype', tf.float32, 'Datatype to use.')
    # None value will make FProp use dtype instead of fprop_dtype.
    # TODO(lepikhin): all @function.Defun should use p.fprop_dtype if it is set.
    p.Define('fprop_dtype', None, 'Activations datatype to use.')
    p.Define(
        'random_seed', None, 'Random seed for deterministic unittests. This '
        'is inherited by child layers if they do not set a random_seed.')
    p.Define('vn', DefaultVN(), 'How variational noise should be applied.')
    p.Define('params_init', py_utils.DefaultParamInit(),
             'How params should be initialized.')
    # is_eval is used to generate graph for eval purpose, typically
    # the eval graph is forward pass of training graph without
    # regularization, e.g. dropout.
    p.Define('is_eval', None, 'True if in eval mode.')
    # In addition to is_eval, also makes additional alterations for graphs
    # being used for inference.
    p.Define('is_inference', None, 'True if in inference mode.')
    # In addition to is_eval/is_inference, indicate that the inference graph is
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
    return p

  @staticmethod
  def CopyBaseParams(from_params, to_params):
    """Copies BaseLayer params from `from_params` to `to_params`."""
    assert issubclass(from_params.cls, BaseLayer)
    assert issubclass(to_params.cls, BaseLayer)
    # Copy-over the BaseLayer params.
    if to_params.dtype == tf.float32:
      to_params.dtype = from_params.dtype
    if from_params.fprop_dtype is not None:
      to_params.fprop_dtype = from_params.fprop_dtype
    if to_params.random_seed is None:
      to_params.random_seed = from_params.random_seed
    if to_params.is_eval is None:
      to_params.is_eval = from_params.is_eval
    if to_params.is_inference is None:
      to_params.is_inference = from_params.is_inference
    if to_params.allow_implicit_capture is None:
      to_params.allow_implicit_capture = from_params.allow_implicit_capture
    if to_params.skip_lp_regularization is None:
      to_params.skip_lp_regularization = from_params.skip_lp_regularization

    # Only copy from base when vn config is using the default setting.
    if to_params.vn == DefaultVN():
      to_params.vn = from_params.vn.Copy()

    # TODO(rpang): derive to_params.params_init.seed from
    # from_params.params_init.seed if it is specified in 'from_params' and not
    # in 'to_params'.
    if py_utils.IsDefaultParamInit(to_params.params_init):
      # Copy over params_init as well.
      to_params.params_init = from_params.params_init.Copy()
    return to_params

  def __init__(self, params):
    """Layer constructor.

    Sub-classes of BaseLayer should decorator its __init__ with
    @base_layer.initializer

    Args:
      params: A params used to construct this layer.
    """
    assert params.name, (
        'Layer params for %s must have a "name"' % self.__class__.__name__)
    self._params = params.Copy()
    tf.logging.debug('Creating layer %s with params: \n %s \n',
                     self.__class__.__name__, str(params))
    # Vars created by this layer.
    self._private_vars = py_utils.NestedMap()
    # Theta derived from this layer's vars.
    self._private_theta = py_utils.NestedMap()
    # Child layers created by this layer through CreateChild/CreateChildren.
    self._private_children = py_utils.NestedMap()
    # Child layers created by this layer. A well-formed layer should
    # have self._private_children equals to self._children_list. I.e.,
    # all child layers are created using CreateChild/CreateChildren.
    self._children_list = []
    # Extra theta's not directly correpond to any underlying vars. For example,
    # the concatenated sharded variables.
    self._extra_theta = py_utils.NestedMap()
    # All registered accumulators.
    self._private_accumulators = py_utils.NestedMap()
    # Layer-private functions. Add with AddFunction.
    self._private_fns = dict()

    self.AddExtraTheta('global_step', py_utils.GetGlobalStep())

  def FPropDefaultTheta(self, *args, **kwargs):
    """Calls `FProp`."""
    return self.FProp(self.theta, *args, **kwargs)

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
        meta = p.cls.FPropMeta(p, tf.TensorShape([128, 20, 50, 32]))

    `meta.flops` gives an estimate count of floating point operations done by
    one `FProp` given an input tensor of shape [128, 20, 50, 32].
    `meta.out_shapes` is a tuple of tensor shapes, which tells you what shape
    of tensors this layer will return.

    Args:
      params: The param of a layer of this layer type.
      *args: Corresponds to FProp with Tensors replaced by `TensorShape`.
      **kwargs: Corresponds to FProp with Tensors replaced by `TensorShape`.

    Returns:
      A `.NestedMap` with

      - flops - The estimated number of floating point operations incurred by
        this fprop.
      - out_shapes - A tuple of `tf.TensorShape`. I.e., `out_shapes[i]`
        represents the shape of the `i`-th returned tensor of the fprop.
    """
    raise NotImplementedError('FPropMeta of %s' % cls)

  @property
  def params(self):
    """Returns the params upon which this layer is built."""
    return self._params

  @property
  def cluster(self):
    """Returns the current cluster configuration."""
    return cluster_factory.Current()

  @property
  def children(self):
    """Returns children layers of this layer in a `.NestedMap`."""
    return self._private_children

  def __getattr__(self, name):
    """Returns the child layer of the given name."""
    if name in self._private_children:
      return self._private_children[name]
    elif (hasattr(type(self), name) and
          isinstance(getattr(type(self), name), property)):
      # There was an AttributeError raised by a property getter.
      # Call property getter again directly to raise the same error.
      return getattr(type(self), name).fget(self)
    else:
      raise AttributeError('%s is not a sub-layer of %s.' % (name, self))

  def GetDescendant(self, path):
    """Returns a descendent layer given the path.

    Args:
      path: a comma separated string denoting a descendant of this layer.

    Returns:
      The descendent layer.
    """
    sub = self
    if path:
      for k in path.split('.'):
        sub = sub.children[k]
    return sub

  @property
  def vars(self):
    """Returns variables of this layer and its children in a `.NestedMap`."""
    ret = self._private_children.Transform(lambda x: x.vars)
    for k in self._private_vars.keys():
      ret[k] = self._private_vars[k]
    return ret

  @property
  def theta(self):
    """Returns theta of this layer and its children in a `.NestedMap`."""
    ret = self._private_children.Transform(lambda x: x.theta)

    private_theta = self._private_theta

    if (self._params.fprop_dtype is not None and
        self._params.fprop_dtype != self._params.dtype):

      def MaybeCastToFPropDtype(x):
        if x.dtype == self._params.dtype:
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
    for k, acc in six.iteritems(self._private_accumulators):
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

  def _CheckName(self, name):
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
        (name, self._private_accumulators.keys()))

  def _VariableCollections(self):
    return [LAYER_WT, '%s_vars' % (self.__class__.__name__)]

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

  def CreateVariable(self, name, var_params, theta_fn=None, *args, **kwargs):
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
            theta_fn=self.AddGlobalVN)

    Args:
      name: Variable name which is used as the key into vars/theta.
      var_params: `Params` used to create the variable.
      theta_fn: A python function that takes a variable's value and returns a
        new value to be used later for computation. Its signature must be
        (tf.Tensor) -> (tf.Tensor).
      *args: List of args passed to `.py_utils.CreateVariable`.
      **kwargs: Keyword args passed to `.py_utils.CreateVariable`.
    """
    self._CheckName(name)
    if (self.params.skip_lp_regularization and
        py_utils.SKIP_LP_REGULARIZATION not in var_params.collections):
      var_params = py_utils.WeightParams(
          shape=var_params.shape,
          dtype=var_params.dtype,
          init=var_params.init,
          collections=(var_params.collections +
                       [py_utils.SKIP_LP_REGULARIZATION]))
    value, var = py_utils.CreateVariable(name, var_params, *args, **kwargs)
    self._private_vars[name] = var
    if theta_fn is not None:
      value = theta_fn(value)
    self._private_theta[name] = value

  def AddExtraTheta(self, theta_name, theta_value):
    """Add extra `theta` that doesn't directly correspond to `vars`."""
    self._CheckName(theta_name)
    self._private_theta[theta_name] = theta_value
    self._extra_theta[theta_name] = theta_value

  def AddGlobalVN(self, value):
    return py_utils.AddGlobalVN(self.params, value)

  def CreateChild(self, name, params):
    """Create a sub layer.

    The created sub layer can be accessed by `name`. E.g.::

        self.CreateChild('foo', ...)
        self.foo.FProp...

    or::

        self.children['foo'].Fprop...
        self.children.foo.Fprop...

    Args:
      name: Sub layer name which is used as the key into vars/theta.
      params: `Hyperparams` object to instantiate a layer.
    """
    self._CheckName(name)
    if not params.name:
      params.name = name
    p = self.CopyBaseParams(self.params, params.Copy())
    child = p.cls(p)
    self._private_children[name] = child

  def CreateChildren(self, name, params_list, child_scopes=None):
    """Create a list of sub layers.

    The created sub layer list can be accessed by `name`. E.g.::

        self.CreateChildren('foo', ...)
        self.foo[10].FProp...

    or::

        self.children['foo'][10].Fprop...
        self.children.foo[10].Fprop...

    Args:
      name: The name for the sub layers, which is used as the key
        into vars/theta.
      params_list: `Hyperparams` objects to instantiate a list of layers.
      child_scopes: If not none, a variable_scope to set for each child.
    """
    self._CheckName(name)

    def CreateChildrenHelper(params_list, child_scopes):
      """Helper to create children recursively."""
      if child_scopes and len(child_scopes) != len(params_list):
        raise ValueError('child_scopes must be same structure as params_list.')
      children = []
      for i, p in enumerate(params_list):
        if isinstance(p, list):
          children.append(
              CreateChildrenHelper(p,
                                   child_scopes[i] if child_scopes else None))
        else:
          p = self.CopyBaseParams(self.params, p.Copy())
          if not p.name:
            p.name = '%s_%d' % (name, i)
          if child_scopes:
            with tf.variable_scope(child_scopes[i]):
              children.append(p.cls(p))
          else:
            children.append(p.cls(p))
      return children

    self._private_children[name] = CreateChildrenHelper(params_list,
                                                        child_scopes)

  def AddChild(self, name, child):
    """Add an existing layer as a sublayer."""
    assert isinstance(child, BaseLayer)
    self._CheckName(name)
    self._private_children[name] = child

  def AddChildren(self, name, children):
    """Add existing layers as sublayers."""
    for child in children:
      assert isinstance(child, BaseLayer)
    self._CheckName(name)
    self._private_children[name] = children

  def _AutoAddChild(self, child):
    """Record that a layer `child` is instantiated by this layer.

    This is a method only called by `base_layer.initializer` decorator.
    Subclasses should not call this method.

    Args:
      child: A sub-layer of this layer.
    """
    self._children_list.append(child)

  def _CheckInvariants(self):
    self._VerifyChildren()
    self._VerifyVarsAndTheta()

  def _VerifyChildren(self):
    """Verify all children created by this layer are via `CreateChild(ren)`."""

    def FindCreatedChildren(parents):
      created_children = []
      for v in parents:
        if isinstance(v, (tuple, list)):
          created_children.extend(FindCreatedChildren(v))
        else:
          created_children.append(v)
      return created_children

    created_children = FindCreatedChildren(
        list(self._private_children.values()))
    for v in self._children_list:
      assert v in created_children, (
          '%s is not created by BaseLayer.CreateChild(ren) in %r.' %
          (v.params.name, self))

  def _VerifyVarsAndTheta(self):
    """Verify that vars and theta have the same nested structure."""

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

  def PostTrainingStepUpdate(self, global_step):
    """Returns a TF op which will be invoked at each training step.

    Subclasses of `BaseLayer` can implement this method. The method should
    return a TF op to be invoked during training after gradients are applied.

    Args:
      global_step: the global step.
    """
    update_ops = [
        child.PostTrainingStepUpdate(global_step)
        for child in self._private_children.Flatten()
    ]
    return tf.group(*update_ops)
