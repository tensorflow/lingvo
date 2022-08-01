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
"""Utilities for model quantization."""

import enum
from typing import Iterable, Optional

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import summary_utils
import numpy as np


@enum.unique
class QDistribution(str, enum.Enum):
  """Distribution type for the inputs for QDomain.

  Symmetric and positive represent general signed and unsigned inputs, while
  padding, relu, ... allow for quantization using known ranges.
  """
  # General distributions.
  SYMMETRIC = 'symmetric'
  POSITIVE = 'positive'
  # Specific activation distributions.
  LOG_SOFTMAX = 'log_softmax'
  PADDING = 'padding'
  RANDOM_UNIFORM = 'random_uniform'
  RELU = 'relu'
  RELU6 = 'relu6'
  SIGMOID = 'sigmoid'
  SOFTMAX = 'softmax'
  TANH = 'tanh'

  @classmethod
  def IsPositive(cls, dist: 'QDistribution') -> bool:
    return dist in [
        QDistribution.POSITIVE,
        QDistribution.PADDING,
        QDistribution.RANDOM_UNIFORM,
        QDistribution.RELU,
        QDistribution.RELU6,
        QDistribution.SIGMOID,
        QDistribution.SOFTMAX,
        QDistribution.TANH,
    ]


class QuantizableLayer(base_layer.BaseLayer):
  """A layer that supports various forms of quantization.

  It is always safe to extend QuantizableLayer instead of BaseLayer (i.e. at
  the base of layer inheritance hierarchies) if any layer in the hierarchy
  may be quantized. Unless if configured/used, all quantization behavior
  is disabled by default.

  Most quantization strategies employed at training time fall into the
  "fake quantization" category, where we add various constraints in the
  forward propagation to quantify and simulate the effect of quantization.
  Within that, we have two major approaches:

    - Active clipping: Usually via a schedule, tensors are actively
      clipped to fall into ranges that we know apriori that the model should
      be able to deal with.
    - Passive tracking and simulation: Passively track the min/max ranges
      of tensors and insert special ops at training and eval time that
      constrain to those ranges.

  The tensors of interest for both approaches are top-level inputs (or
  embeddings), outputs of arithmetic operations (add, mul, tanh, etc) and
  weights. While the actual process of quantizing can be quite complex and
  involve an end to end view of the system, from a modeling perspective, it
  can be thought of as providing tags/decorators to arithmetic inputs/outputs.
  It would be appropriate to think of these as casts which alter the way that
  the arithmetic operation is tracked and quantize (if Python/Tensorflow were
  a more strongly typed environment, they would indeed represent types in the
  type system but given the loose typing, it is just an honor system).

  The "decorators" are:

  - QWeight: Tags a weight (typically a tf.Variable) as a weight quantized type.
  - QRAct(act, dist: QDistribution): Tags an activation with a known or
    configurable quantization range.
  - QAct: Tags an activation as a generic quantized intermediate value.
    These are also tagged with a layer-unique name. All QActs with the
    same name will be considered the same from a numerical range/precision
    perspective.

  Tagging things in this way allows us to, via hyperparameters, associate
  one or more quantization domains (QDomain) with the layer that will
  actually perform the necessary tracking and transformations needed at
  training and inference time to ensure that the layer can operate in low
  bit inference engines that only have quantized numeric representations.
  See the SampleQuantizedProjectionLayer in the unit test for an example layer
  that has had these tags applied.

  As a note on terminology, domain/QDomain here refers to a distinct set of
  quantization rules to apply to a subset of tensors. Most layers will only
  have one QDomain (default). The concept exists for layers which have been
  specially created to operate in more than one quantized precision (i.e. an
  RNN cell that uses 8bit quantization for inputs/outputs and 16bit
  quantization for internal state arithmetic). Such uses should be rare.

  **Convenience functions:**

  The layer adds a number of convenience functions to the layer's 'fns'
  function library. These mirror similarly named functions in TensorFlow but
  automatically add the necessary annotations. All such functions take the
  following named parameters:

    - qout_name: Name of QAct (setup with TrackQActs) for dynamic range
      tracking.

  Functions that have a natural output range will have default values for
  qmin/qmax so that they just work. Functions that do not have a natural
  output range must use qout_name.

  Natural or configurable range functions

  - qtanh
  - qsigmoid
  - qsoftmax
  - qrelu  (quantized to [0, 1])
  - qrelu6
  - qlogsoftmax  (configured in FakeQDomain.Params)
  - qrandom_uniform

  Dynamic range functions:

  - qadd
  - qsubtract
  - qmultiply
  - qlog

  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('qdomain', hyperparams.Params(),
             'Container for quantization domains.')
    p.qdomain.Define('default', None, 'Default quantization domain.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    # A set of all tracked activation and weight names.
    self._all_names = set()
    self._act_name_to_qdomain_name = dict()
    self._weight_name_to_qdomain_name = dict()

    # Instantiate quantization domains.
    self._qdomains = dict()  # Dict of qdname -> QDomain or None
    for qdname in dir(p.qdomain):
      qdparams = p.qdomain.Get(qdname)
      if qdparams is None:
        self._qdomains[qdname] = None
      else:
        if not issubclass(qdparams.cls, QDomain):
          raise TypeError(f'Expected p.qdomain.{qdname} to extend QDomain, but '
                          f'got {qdparams.cls}')
        qdchild_name = 'qdomain_' + qdname
        self.CreateChild(qdchild_name, qdparams)
        self._qdomains[qdname] = self.children[qdchild_name]
    self._AddQuantizationFunctions()

  def _child_variable_scope_override(self):
    p = self.params
    res = super()._child_variable_scope_override()
    for qdname in dir(p.qdomain):
      qdparams = p.qdomain.Get(qdname)
      if qdparams is None:
        continue
      qdchild_name = 'qdomain_' + qdname
      res[qdchild_name] = [p.name + '/q']
    return res

  def TrackQActs(self,
                 *act_names: str,
                 shape: Optional[Iterable[int]] = None,
                 feature_axes: Optional[Iterable[int]] = None,
                 domain: str = 'default'):
    """Track activations and create variables for quantization as needed.

    The activations will be associated with a specific named QDomain. Most
    layers are simple enough to only require a single 'default' QDomain.
    Additional QDomains can be defined as parameters to control fine grained
    aspects of quantization.

    Args:
      *act_names: A sequence of names to use to track statistics for layer
        activations.
      shape: The optional shape of the activation(s) to track. Can be used for
        static per-channel activation quantization.
      feature_axes: The optional feature axes of the activation(s) to track. Can
        be used for static per-channel activation quantization.
      domain: The name of the QDomain to use to for quantization.
    """
    for act_name in act_names:
      if act_name in self._all_names:
        raise ValueError(
            f"act_name='{act_name}' is already tracked for this layer.")
      self._all_names.add(act_name)
      self._act_name_to_qdomain_name[act_name] = domain
    qd = self._GetQDomain(domain)
    if qd is not None:
      qd.TrackQActs(*act_names, shape=shape, feature_axes=feature_axes)

  def TrackQWeight(self,
                   weight_name,
                   shape,
                   feature_axis,
                   domain: str = 'default',
                   *,
                   tensor_split_dims_mapping=None,
                   device_mesh=None,
                   legacy_aqt_weight_name=None):
    """Creates Quantized weights for later use.

    Weight that will later be quantized must be created first, preferably
    in _CreateLayerVariables().

    Args:
      weight_name: Positional parameters are taken to be weight names to create.
      shape: Shape of the weight.
      feature_axis: axis corresponding to output channel/feature for weights.
      domain: Name of the QDomain to use for quantization.
      tensor_split_dims_mapping: A list of integers that map each tensor axis
        to the device mesh axis along which it is sharded.
      device_mesh: A numpy.ndarray describing the topology of a device mesh to
        partition the created variable onto.
      legacy_aqt_weight_name: Used for compatibility with old checkpoints.
    """
    if weight_name in self._all_names:
      raise ValueError(
          f"weight_name='{weight_name}' is already tracked for this layer.")
    self._all_names.add(weight_name)
    self._weight_name_to_qdomain_name[weight_name] = domain
    qd = self._GetQDomain(domain)
    if qd is not None:
      # We're calling child.CreateVariable() here rather than
      # self.CreateVariable(), which messes up the automatic variable scope
      # handing, so we need to set some manual variable scopes for backwards
      # compatibility.
      for qdname in self._qdomains:
        if qd is self._qdomains[qdname]:
          qdchild_name = 'qdomain_' + qdname
          with self._CreateChildContext(qdchild_name):
            with tf.variable_scope(qd.params.name):
              qd.TrackQWeight(weight_name, shape, feature_axis,
                              tensor_split_dims_mapping, device_mesh,
                              legacy_aqt_weight_name)
          break

  def QAct(self, act_name, act, eval_only=False):
    """Quantizes an activation.

    act_name must have been previously created via TrackQActs.

    Args:
      act_name: Previously created act_name to quantize to.
      act: Activation to quantize.
      eval_only: Whether to only apply quantization pressure at eval time.

    Returns:
      The tensor, quantized.
    """
    if act_name not in self._act_name_to_qdomain_name:
      raise ValueError(
          f"The given act_name='{act_name}' must first be tracked using "
          f'TrackQActs. Expected one of {self._act_name_to_qdomain_name.keys()}'
      )
    qd = self._GetQDomain(self._act_name_to_qdomain_name[act_name])
    if not qd:
      return act
    else:
      return qd.QuantizeAct(act_name, act, eval_only=eval_only)

  def QWeight(self, w, domain: str = 'default'):
    """Quantizes a weight.

    Args:
      w: The weight tensor.
      domain: Name of the QDomain to use for quantization.
    Returns:
      The weights quantized.
    """
    qd = self._GetQDomain(domain)
    return qd.QuantizeWeight(w) if qd else w

  def _ValidateArgName(self, op_arg_name: str, name: Optional[str]):
    if name is not None and name not in self._all_names:
      raise ValueError(
          f"Expected {op_arg_name}='{name}' to be None or one of "
          f'{self._all_names}. Use TrackQActs or TrackQWeight to create it')

  def QMatmul(self,
              lhs,
              rhs,
              *,
              lhs_name: Optional[str] = None,
              rhs_name: Optional[str] = None,
              lhs_dist: QDistribution = QDistribution.SYMMETRIC,
              rhs_dist: QDistribution = QDistribution.SYMMETRIC,
              ensure2d: bool = False,
              qdomain: str = 'default',
              **op_kwargs):
    self._ValidateArgName('lhs_name', lhs_name)
    self._ValidateArgName('rhs_name', rhs_name)
    qd = self._GetQDomain(qdomain)
    if qd is None:
      if ensure2d:
        return py_utils.Matmul(lhs, rhs, **op_kwargs)
      else:
        return tf.matmul(lhs, rhs, **op_kwargs)
    else:
      return qd.QMatmul(
          lhs,
          rhs,
          lhs_name=lhs_name,
          rhs_name=rhs_name,
          lhs_dist=lhs_dist,
          rhs_dist=rhs_dist,
          ensure2d=ensure2d,
          **op_kwargs)

  def QConv1D(self,
              inputs,
              filters,
              strides,
              padding,
              *,
              inputs_name: Optional[str] = None,
              filters_name: Optional[str] = None,
              inputs_dist: QDistribution = QDistribution.SYMMETRIC,
              filters_dist: QDistribution = QDistribution.SYMMETRIC,
              qdomain: str = 'default',
              **op_kwargs):
    self._ValidateArgName('inputs_name', inputs_name)
    self._ValidateArgName('filters_name', filters_name)
    qd = self._GetQDomain(qdomain)
    if qd is None:
      return tf.nn.conv1d(inputs, filters, strides, padding, **op_kwargs)
    else:
      return qd.QConv1D(
          inputs,
          filters,
          strides,
          padding,
          inputs_name=inputs_name,
          filters_name=filters_name,
          inputs_dist=inputs_dist,
          filters_dist=filters_dist,
          **op_kwargs)

  def QConv2D(self,
              inputs,
              filters,
              strides,
              padding,
              *,
              inputs_name: Optional[str] = None,
              filters_name: Optional[str] = None,
              inputs_dist: QDistribution = QDistribution.SYMMETRIC,
              filters_dist: QDistribution = QDistribution.SYMMETRIC,
              is_depthwise: bool = False,
              qdomain: str = 'default',
              **op_kwargs):
    self._ValidateArgName('inputs_name', inputs_name)
    self._ValidateArgName('filters_name', filters_name)
    qd = self._GetQDomain(qdomain)
    if qd is None:
      if is_depthwise:
        return tf.nn.depthwise_conv2d(inputs, filters, strides, padding,
                                      **op_kwargs)
      return tf.nn.conv2d(inputs, filters, strides, padding, **op_kwargs)
    else:
      return qd.QConv2D(
          inputs,
          filters,
          strides,
          padding,
          inputs_name=inputs_name,
          filters_name=filters_name,
          inputs_dist=inputs_dist,
          filters_dist=filters_dist,
          is_depthwise=is_depthwise,
          **op_kwargs)

  def QEinsum(self,
              equation,
              lhs,
              rhs,
              *,
              lhs_name: Optional[str] = None,
              rhs_name: Optional[str] = None,
              lhs_dist: QDistribution = QDistribution.SYMMETRIC,
              rhs_dist: QDistribution = QDistribution.SYMMETRIC,
              qdomain: str = 'default',
              **einsum_kwargs):
    self._ValidateArgName('lhs_name', lhs_name)
    self._ValidateArgName('rhs_name', rhs_name)
    qd = self._GetQDomain(qdomain)
    if qd is None:
      return tf.einsum(equation, lhs, rhs, **einsum_kwargs)
    else:
      return qd.QEinsum(
          equation,
          lhs,
          rhs,
          lhs_name=lhs_name,
          rhs_name=rhs_name,
          lhs_dist=lhs_dist,
          rhs_dist=rhs_dist,
          **einsum_kwargs)

  def _ValidateWeight(self, w_name, w=None):
    if w_name not in self._weight_name_to_qdomain_name:
      raise ValueError(
          f"The given w_name='{w_name}' must first be tracked using "
          'TrackQWeight. Expected one of '
          f'{self._weight_name_to_qdomain_name.keys()}')

    # If w is a variable on this layer, then validate that w_name matches the
    # name given to w in self.CreateVariable.
    if w is not None and hasattr(w, 'ref') and w.ref() in self.vars:
      expected_name = w.name.split('/')[-2]  # model/path/layer/weight/var:0
      if w_name != expected_name:
        raise ValueError(
            f'Expected the AQT weight name to match the name of non-AQT '
            f"weight, but got '{w_name}' and '{expected_name}'."
            f'\nFull weight info:\n  {w}')

  def ToAqtWeight(self, w_name, w, feature_axis, expected_scale_shape=None):
    """Quantized integer weight AQT style.

    This only scales, rounds and clips; resulting quantized weight would be
    either integer or integer emulated in float.

    w_name must have been previously created via TrackQWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      w: The weight tensor.
      feature_axis: axis corresponding to output channel/feature for weights.
      expected_scale_shape: Optional shape to verify if scale shape is expected.
        Defaults to None.

    Returns:
      Quantized weights.
    """
    self._ValidateWeight(w_name, w)
    qd = self._GetQDomain(self._weight_name_to_qdomain_name[w_name])
    if not qd:
      return w
    return qd.ToAqtWeight(
        w_name,
        w,
        feature_axis=feature_axis,
        expected_scale_shape=expected_scale_shape)

  def FromAqtWeight(self, w_name, out, merge_feature_axes=False):
    """Rescales the output corresponding to AQT style quantized matmul's weight.

    Uses the same scale used by `ToAqtWeight` and apply its inverse to rescale.

    w_name must have been previously created via TrackQWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      out: The tensor to rescale.
      merge_feature_axes: whether or the feature axes have been reshaped into a
        single axis in 'out'.

    Returns:
      Rescaled output.
    """
    self._ValidateWeight(w_name)
    qd = self._GetQDomain(self._weight_name_to_qdomain_name[w_name])
    return qd.FromAqtWeight(w_name, out) if qd else out

  def ToAqtConv(self,
                w_name,
                act,
                weight,
                w_feature_axis,
                act_distribution=QDistribution.SYMMETRIC,
                w_expected_scale_shape=None):
    """Quantizes Weights and activations for convolutions.

    Refer to quantizable_layer.ToAqtConv.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      act: The activation tensor to quantize.
      weight: The weight tensor to quantizes.
      w_feature_axis: axis corresponding to output channel/feature for weights.
      act_distribution: Distribution of act.
      w_expected_scale_shape: Optional shape to verify if scale shape is
        expected. Defaults to None.

    Returns:
      Quantized act and weight.
    """
    self._ValidateWeight(w_name, weight)
    qd = self._GetQDomain(self._weight_name_to_qdomain_name[w_name])
    if not qd:
      return act, weight
    return qd.ToAqtConv(
        w_name,
        act=act,
        weight=weight,
        act_distribution=act_distribution,
        w_feature_axis=w_feature_axis,
        w_expected_scale_shape=w_expected_scale_shape)

  def FromAqtConv(self, w_name, output, *, is_depthwise=False):
    """Rescales the output corresponding to AQT quantized convolution.

    Refer to quantizable_layer.FromAqtConv.

    Args:
      w_name: weight name.
      output: The tensor to rescale.
      is_depthwise: Whether or not this follows a DepthwiseConv, which merges
        the feature axes in the output tensor.

    Returns:
      Rescaled output.
    """
    self._ValidateWeight(w_name)
    qd = self._GetQDomain(self._weight_name_to_qdomain_name[w_name])
    if qd is None:
      return output
    return qd.FromAqtConv(w_name, output, is_depthwise=is_depthwise)

  def ToAqtInputs(self,
                  w_name,
                  act,
                  weight,
                  w_feature_axis,
                  act_distribution=QDistribution.SYMMETRIC,
                  w_expected_scale_shape=None):
    """Quantizes weights and activations for (act * w) matmul AQT style.

    This only scales, rounds and clips; resulting quantized inputs would be
    either integer ot integer emulated in float.

    w_name must have been previously created via TrackQWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      act: The activation tensor to quantize.
      weight: The weight tensor to quantizes.
      w_feature_axis: axis corresponding to output channel/feature for weights.
      act_distribution: Distribution of act.
      w_expected_scale_shape: Optional shape to verify if scale shape is
        expected. Defaults to None.

    Returns:
      Quantized act and weight.
    """
    self._ValidateWeight(w_name, weight)
    qd = self._GetQDomain(self._weight_name_to_qdomain_name[w_name])
    if not qd:
      return act, weight
    return qd.ToAqtInputs(
        w_name,
        act=act,
        weight=weight,
        act_distribution=act_distribution,
        w_feature_axis=w_feature_axis,
        w_expected_scale_shape=w_expected_scale_shape)

  def FromAqtMatmul(self, w_name, output):
    """Rescales the output corresponding to AQT style quantized matmul.

    Uses the same scales used by `ToAqtInputs` and apply its inverse to rescale.

    w_name must have been previously created via TrackQWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      output: The tensor to rescale.

    Returns:
      Rescaled output.
    """
    self._ValidateWeight(w_name)
    qd = self._GetQDomain(self._weight_name_to_qdomain_name[w_name])
    return qd.FromAqtMatmul(w_name, output) if qd else output

  def ToAqtActActInputs(self,
                        act_lhs,
                        act_rhs,
                        *,
                        act_lhs_distribution=QDistribution.SYMMETRIC,
                        act_rhs_distribution=QDistribution.SYMMETRIC,
                        domain: str = 'default'):
    """Quantizes activations for (act * act) matmul AQT style.

    This only scales, rounds and clips; resulting quantized acts would be
    either integer or integer emulated in float.

    Args:
      act_lhs: Left hand side activation.
      act_rhs: Right hand side activation.
      act_lhs_distribution: Distribution of act_lhs.
      act_rhs_distribution: Distribution of act_rhs.
      domain: Custom domain to match (defaults to 'default').

    Returns:
      Quantized activations corresponding to act_lhs and act_rhs.
    """
    qd = self._GetQDomain(domain)
    if not qd:
      return act_lhs, act_rhs

    return qd.ToAqtActActInputs(
        act_lhs=act_lhs,
        act_rhs=act_rhs,
        act_lhs_distribution=act_lhs_distribution,
        act_rhs_distribution=act_rhs_distribution)

  def FromAqtActActMatmul(self, output, domain: str = 'default'):
    """Rescales the output of (act*act) matmul for AQT style quantized acts.

    Args:
      output: Previously created w_name QWeight to quantize weight.
      domain: Custom domain to match (defaults to 'default').

    Returns:
      Rescaled output.
    """
    qd = self._GetQDomain(domain)
    if not qd:
      return output

    return qd.FromAqtActActMatmul(output)

  def _GetQDomain(self, domain: str):
    """Gets the QDomain matching a given domain name.

    Args:
      domain: User specified domain name.

    Returns:
      The requested QDomain, the 'default' QDomain or None.
    """
    qd = self._qdomains[domain]
    if qd is not None:
      return qd
    else:
      return self._qdomains['default']

  def GetQDomainParams(self, domain: str):
    """Gets domain's Params if they're set, and the default Params otherwise.

    Args:
      domain: User specified domain name.

    Returns:
      Params for the QDomain matching the requested domain name if they're set,
      and params for the 'default' QDomain otherwise. Will return None if both
      p.qdomain.domain and p.qdomain.default are None.
    """
    p = self.params
    qdparams = p.qdomain.Get(domain)
    if qdparams is None:
      qdparams = p.qdomain.default
    return qdparams

  def QRAct(self, act, dist: QDistribution, domain: str = 'default'):
    """Quantizes act according to its distribution."""
    qd = self._GetQDomain(domain)
    return act if qd is None else qd.QRAct(act, dist)

  def _AddQuantizationFunctions(self):
    """Adds standard quantization functions against the given layer."""

    def WrapOp(op_name, op, dist=None):
      """Adds a wrapper op to the layer's fns."""

      def Wrapped(*op_args,
                  qout_name=None,
                  qdomain: str = 'default',
                  **op_kwargs):
        """Wraps a native op."""
        if qout_name is None and dist is None:
          raise ValueError(
              f'Quantized op "{op_name}" requires qout_name to be set.')

        # Provide a better default name if none provided.
        if 'name' not in op_kwargs and qout_name is not None:
          op_kwargs['name'] = '%s_%s' % (op_name, qout_name)

        # Invoke original.
        y = op(*op_args, **op_kwargs)

        # Handle the output.
        if qout_name is not None:
          y = self.QAct(qout_name, y)
        else:
          y = self.QRAct(y, dist, qdomain)  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
        return y

      self.AddFunction(op_name, Wrapped)

    # Quantize output activations based on tracked tensors.
    WrapOp('qadd', tf.add)
    WrapOp('qsubtract', tf.subtract)
    WrapOp('qmultiply', tf.multiply)
    WrapOp('qlog', tf.math.log)

    # TODO(shivaniagrawal): delete the following wrappers, as they are
    # duplicated with self.QMatmul, self.QConv.
    WrapOp('qmatmul', py_utils.Matmul)
    WrapOp('qbatchmatmul', tf.matmul)
    WrapOp('qconv1d', tf.nn.conv1d)

    # Quantizing activations based on distribution
    WrapOp('qtanh', tf.tanh, dist=QDistribution.TANH)
    WrapOp('qsigmoid', tf.sigmoid, dist=QDistribution.SIGMOID)
    WrapOp('qsoftmax', tf.nn.softmax, dist=QDistribution.SOFTMAX)
    WrapOp('qlogsigmoid', tf.math.log_sigmoid, dist=QDistribution.LOG_SOFTMAX)
    WrapOp('qlogsoftmax', tf.nn.log_softmax, dist=QDistribution.LOG_SOFTMAX)
    WrapOp('qrelu', tf.nn.relu, dist=QDistribution.RELU)
    WrapOp('qrelu6', tf.nn.relu6, dist=QDistribution.RELU6)
    WrapOp(
        'qrandom_uniform', tf.random.uniform, dist=QDistribution.RANDOM_UNIFORM)


class QDomain(base_layer.BaseLayer):
  """Base class for a quantization domain layer.

  This implementation doubles as a no-op quantization domain.
  """

  def __init__(self, params):
    super().__init__(params)
    self.all_names = set()
    self.act_names = set()
    self.weight_names = set()

  @property
  def bits(self):
    """Retrieves the bits used by this quantization layer.

    Returns:
      The number of bits available to this qdomain or None if unquantized.
    """
    return None

  def QuantizeWeight(self, w):
    """Quantizes a weight.

    Args:
      w: Weight tensor to quantize.

    Returns:
      Quantized weight.
    """
    return w

  def QRAct(self, act, dist: QDistribution):
    del dist
    return act

  def QMatmul(self,
              lhs,
              rhs,
              *,
              lhs_name: Optional[str],
              rhs_name: Optional[str],
              lhs_dist: QDistribution = QDistribution.SYMMETRIC,
              rhs_dist: QDistribution = QDistribution.SYMMETRIC,
              out_name: Optional[str],
              ensure2d: bool = False,
              **op_kwargs):
    del lhs_name, rhs_name, lhs_dist, rhs_dist, out_name
    if ensure2d:
      return py_utils.Matmul(lhs, rhs, **op_kwargs)
    return tf.matmul(lhs, rhs, **op_kwargs)

  def QConv1D(self,
              inputs,
              filters,
              strides,
              padding,
              *,
              inputs_name: Optional[str],
              filters_name: Optional[str],
              inputs_dist: QDistribution = QDistribution.SYMMETRIC,
              filters_dist: QDistribution = QDistribution.SYMMETRIC,
              **op_kwargs):
    del inputs_name, filters_name, inputs_dist, filters_dist
    return tf.nn.conv1d(inputs, filters, strides, padding, **op_kwargs)

  def QConv2D(self,
              inputs,
              filters,
              strides,
              padding,
              *,
              inputs_name: Optional[str],
              filters_name: Optional[str],
              inputs_dist: QDistribution = QDistribution.SYMMETRIC,
              filters_dist: QDistribution = QDistribution.SYMMETRIC,
              is_depthwise: bool = False,
              **op_kwargs):
    del inputs_name, filters_name, inputs_dist, filters_dist
    if is_depthwise:
      return tf.nn.depthwise_conv2d(inputs, filters, strides, padding,
                                    **op_kwargs)
    return tf.nn.conv2d(inputs, filters, strides, padding, **op_kwargs)

  def QEinsum(self,
              equation,
              lhs,
              rhs,
              *,
              lhs_name: Optional[str],
              rhs_name: Optional[str],
              lhs_dist: QDistribution = QDistribution.SYMMETRIC,
              rhs_dist: QDistribution = QDistribution.SYMMETRIC,
              **einsum_kwargs):
    del lhs_name, rhs_name, lhs_dist, rhs_dist
    return tf.einsum(equation, lhs, rhs, **einsum_kwargs)

  def ToAqtConv(self,
                w_name,
                act,
                weight,
                w_feature_axis,
                act_distribution=QDistribution.SYMMETRIC,
                w_expected_scale_shape=None):
    """Quantizes Weights and activations for convolutions.

    Refer to quantizable_layer.ToAqtConv.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      act: The activation tensor to quantize.
      weight: The weight tensor to quantizes.
      w_feature_axis: axis corresponding to output channel/feature for weights.
      act_distribution: Distribution of act.
      w_expected_scale_shape: Optional shape to verify if scale shape is
        expected. Defaults to None.

    Returns:
      Quantized act and weight.
    """
    del w_feature_axis, w_expected_scale_shape, w_name, act_distribution
    return act, weight

  def FromAqtConv(self, w_name, output, *, is_depthwise=False):
    """Rescales the output corresponding to AQT quantized convolution.

    Refer to quantizable_layer.FromAqtConv.

    Args:
      w_name: weight name.
      output: The tensor to rescale.
      is_depthwise: Whether or not this follows a DepthwiseConv, which merges
        the feature axes in the output tensor.

    Returns:
      Rescaled output.
    """
    del w_name, is_depthwise
    return output

  def ToAqtInputs(self,
                  w_name,
                  act,
                  weight,
                  w_feature_axis,
                  act_distribution=QDistribution.SYMMETRIC,
                  w_expected_scale_shape=None):
    """Quantizes weights and activations for (act * w) matmul AQT style.

    Refer to quantizable_layer.ToAqtInputs.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      act: The activation tensor to quantize.
      weight: The weight tensor to quantizes.
      w_feature_axis: axis corresponding to output channel/feature for weights.
      act_distribution: Distribution of act.
      w_expected_scale_shape: Optional shape to verify if scale shape is
        expected. Defaults to None.

    Returns:
      Quantized act and weight.
    """
    del w_feature_axis, w_expected_scale_shape, w_name, act_distribution
    return act, weight

  def FromAqtMatmul(self, w_name, output):
    """Rescales the output corresponding to AQT quantized matmuls.

    Refer to quantizable_layer.FromAqtOutput.

    Args:
      w_name: weight name.
      output: The tensor to rescale.

    Returns:
      Rescaled output.
    """
    del w_name
    return output

  def ToAqtWeight(self, w_name, w, feature_axis, expected_scale_shape):
    """Quantized weight AQT style.

    Refer to quantizable_layer.ToAqtWeight.

    Args:
      w_name: weight name.
      w: The weight tensor.
      feature_axis: axis corresponding to output channel/feature for weights.
      expected_scale_shape: Optional shape to verify if scale shape is expected.

    Returns:
      Quantized weights.
    """
    del feature_axis, expected_scale_shape, w_name
    return w

  def FromAqtWeight(self, w_name, out, merge_feature_axes=False):
    """Rescales the output corresponding to AQT quantized matmuls' weight.

    Refer to quantizable_layer.FromAqtWeight.

    Args:
      w_name: weight name.
      out: The tensor to rescale.
      merge_feature_axes: whether or the feature axes have been reshaped into a
        single axis in 'out'.

    Returns:
      Rescaled output.
    """
    del w_name
    return out

  def ToAqtActActInputs(self,
                        act_lhs,
                        act_rhs,
                        act_lhs_distribution=QDistribution.SYMMETRIC,
                        act_rhs_distribution=QDistribution.SYMMETRIC):
    """Quantizes activations for (act * act) matmul AQT style.

    This only scales, rounds and clips; resulting quantized acts would be
    either integer or integer emulated in float.

    Args:
      act_lhs: Left hand side activation.
      act_rhs: Right hand side activation.
      act_lhs_distribution: Distribution of act_lhs.
      act_rhs_distribution: Distribution of act_rhs.

    Returns:
      Quantized activations corresponding to act_lhs and act_rhs.
    """
    del act_lhs_distribution, act_rhs_distribution
    return act_lhs, act_rhs

  def FromAqtActActMatmul(self, output):
    """Rescales output of dynamic matmul (act * act).

    Args:
      output: output, corresponds to tf.matmul(act_lhs, act_rhs)

    Returns:
      Rescaled output.
    """
    return output

  def QuantizeConstantRange(self, t, min_value, max_value):
    """Quantizes a true-constant range that is not used for arithmetic.

    This supports special values like padding that should have a precise
    range that we do not deviate from.

    Args:
      t: Tensor to quantize.
      min_value: Min of the range.
      max_value: Max of the range.

    Returns:
      Quantized tensor.
    """
    return t

  def QuantizeNaturalRange(self, t, min_value, max_value):
    """Quantizes a tensor with a known, natural range.

    Args:
      t: Tensor to quantize.
      min_value: Min value of the range.
      max_value: Max value of the range.

    Returns:
      Quantized tensor.
    """
    return t

  def TrackQActs(self,
                 *act_names: str,
                 shape: Optional[Iterable[int]] = None,
                 feature_axes: Optional[Iterable[int]] = None):
    """Tracks activations and creates variables for quantization as needed."""
    for act_name in act_names:
      if act_name in self.all_names:
        raise ValueError(
            f"act_name '{act_name}' is already tracked for this qdomain.")
      self.all_names.add(act_name)
      self.act_names.add(act_name)

  def TrackQWeight(self,
                   weight_name,
                   shape,
                   feature_axis,
                   mesh_split=None,
                   device_mesh=None,
                   legacy_aqt_weight_name=None):
    """Creates a QWeight with weight_name and given shape.

    Args:
      weight_name: Unique name (within layer) for this weight.
      shape: Expected shape of the weight.
      feature_axis: axis corresponding to output channel/feature for weights.
      mesh_split: A list of integers that map each tensor axis to the device
        mesh axis along which it is sharded.
      device_mesh: A numpy.ndarray describing the topology of a device mesh to
        partition the created variable onto.
      legacy_aqt_weight_name: Used for compatibility with old checkpoints.
    """
    if weight_name in self.all_names:
      raise ValueError(
          f"weight_name '{weight_name}' is already tracked for this qdomain.")
    self.all_names.add(weight_name)
    self.weight_names.add(weight_name)

  def QuantizeAct(self, act_name: str, act: tf.Tensor, eval_only: bool = False):
    """Quantizes a tensor with act_name previously created with TrackQActs.

    Args:
      act_name: Activation name.
      act: Activations to quantize.
      eval_only: Whether to only apply quantization pressure at eval time.

    Returns:
      Quantized tensor.
    """
    return act


class FakeQDomain(QDomain):
  """Base for QDomains using tf.quantization.fake_quant_with_*."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('narrow_to_asym_bit_depth', True,
             "Match TFLite's softmax and tanh bound calculations.")
    p.Define(
        'log_softmax_range', None,
        'Manual range for quantizing logsoftmax activations. Should be '
        'None or a sequence of two numbers.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not (p.log_softmax_range is None or len(p.log_softmax_range) == 2):
      raise ValueError(f'p.log_softmax_range={p.log_softmax_range} should be '
                       'None or a sequence of two numbers')

  def _MaybeNarrowToAsymBitDepth(self, qmin, qmax):
    if self.params.narrow_to_asym_bit_depth:
      qrange = qmax - qmin
      qmax = qmin + qrange * (2**self.bits - 1) / (2**self.bits)
    return qmin, qmax

  def QRAct(self, act, dist: QDistribution):
    p = self.params
    if dist == QDistribution.LOG_SOFTMAX:
      if p.log_softmax_range is None:
        raise ValueError(
            'p.log_softmax_range must be set when using fns.qlogsoftmax '
            'without qout_name or QRAct(..., dist=QDistribution.LOG_SOFTMAX)')
      qmin, qmax = p.log_softmax_range
      return self.QuantizeNaturalRange(act, qmin, qmax)
    if dist == QDistribution.PADDING:
      return self.QuantizeConstantRange(act, 0.0, 1.0)
    elif dist == QDistribution.RELU:
      return self.QuantizeNaturalRange(act, 0.0, 1.0)
    elif dist == QDistribution.RELU6:
      return self.QuantizeNaturalRange(act, 0.0, 6.0)
    elif dist == QDistribution.SIGMOID:
      return self.QuantizeNaturalRange(act, 0.0, 1.0)
    elif dist == QDistribution.SOFTMAX:
      qmin, qmax = self._MaybeNarrowToAsymBitDepth(0.0, 1.0)
      return self.QuantizeNaturalRange(act, qmin, qmax)
    elif dist == QDistribution.TANH:
      qmin, qmax = self._MaybeNarrowToAsymBitDepth(-1.0, 1.0)
      return self.QuantizeNaturalRange(act, qmin, qmax)
    elif dist == QDistribution.RANDOM_UNIFORM:
      return self.QuantizeNaturalRange(act, 0.0, 1.0)
    else:
      raise ValueError(f'cannot quantize act with dist={dist} to know range')


class BaseClippingCapSchedule(base_layer.BaseLayer):
  """Base class for clipping cap schedules."""

  @property
  def is_quantized(self):
    return False

  def GetEndRange(self):
    """Public method to get the final range as a constant.

    Note that this returns the "ideal" end range (i.e. -1..1) as opposed to
    the actual range, which has its upper bound slightly adjusted based on
    the bit depth of the quantized type. In this sense, this value is a lie,
    but it is a consistent lie that can be corrected for downstream by the
    inference engine once it has inferred the actual quantized types being
    used.

    Note that this also assumes the default start/end caps. Some internal
    parts may use altered caps or bit depths.

    Returns:
      Tuple of (min, max) for the final range.
    """
    raise NotImplementedError('Abstract Method: GetEndRange')

  def GetQuantizedEndRange(self):
    """Gets the quantized ending range.

    Unlike GetEndRange(), this takes quantization effects into account.
    The default implementation just returns self.GetEndRange(). Subclasses
    can include additional keyword arguments, tightly coupling them to callers
    of specific types.

    Returns:
      Tuple of (min, max) for the final range.
    """
    assert not self.is_quantized
    return self.GetEndRange()

  def ApplyConstantClip(self, x, min_value, max_value):
    """Applies a constant clip with the clipping op for the implementation.

    This is a special case which allows applying a custom clipping range to
    constants that are not used arithmetically. This exists to support padding.

    Args:
      x: Tensor to clip.
      min_value: Minimum value.
      max_value: Maximum value.
    Returns:
      Tensor clipped.
    """
    raise NotImplementedError('Abstract method: ApplyConstantClip')

  def GetState(self, theta):
    """Gets a state tensor that can be used to calculate clipping.

    The state will be a float32 tensor that is safe to pass to TF functions.

    Args:
      theta: Layer theta.
    Returns:
      An opaque tensor to be passed to ApplyClippingWithState().
    """
    raise NotImplementedError('Abstract method: GetState')

  def ApplyClipping(self, theta, x, **kwargs):
    """Applies clipping to x.

    Args:
      theta: Layer theta.
      x: Input tensor to clip.
      **kwargs: Additional implementation specific kwargs.
    Returns:
      Clipped (or identity) x.
    """
    return self.ApplyClippingWithState(self.GetState(theta), x, **kwargs)

  def ApplyClippingWithState(self, state, x):
    """Applies clipping to x.

    Args:
      state: A previously obtained value of GetState().
      x: Input tensor to clip.
    Returns:
      Clipped (or identity) x.
    """
    raise NotImplementedError('Abstract Method: ApplyClippingWithState')


class IdentityClippingCapSchedule(BaseClippingCapSchedule):
  """Dummy cc schedule (useful in some cases instead of None)."""

  def GetEndRange(self):
    np_dtype = self.params.dtype.as_numpy_dtype
    np_info = np.finfo(np_dtype)
    return (np_info.min, np_info.max)

  def ApplyConstantClip(self, x, min_value, max_value):
    return x

  def GetState(self, theta):
    return tf.zeros([1], tf.float32)

  def ApplyClippingWithState(self, state, x):
    return x


class LinearClippingCapSchedule(BaseClippingCapSchedule):
  """Class for linear clipping cap decay."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('start_step', 0,
             'We start gradually narrowing clipping cap from start_step.')
    p.Define('end_step', 15000,
             'We reach end_cap by end_step.')
    p.Define('start_cap', 8.0,
             'We gradually narrow the clipping range over the course of '
             'training. This is the clipping range we apply when training '
             'starts.')
    p.Define('end_cap', 1.0,
             'We gradually narrow the clipping range over the course of '
             'training. This is the clipping range we apply towards the end '
             'of training.')
    p.name = 'CCSchedule'
    return p

  @property
  def is_quantized(self):
    return False

  def ApplyConstantClip(self, x, min_value, max_value):
    return tf.clip_by_value(x, min_value, max_value)

  def GetState(self, theta):
    return self._Value()

  def ApplyClippingWithState(self, state, x):
    """Applies clipping to x.

    Args:
      state: Clipping state.
      x: Input tensor to clip.
    Returns:
      Clipped (or identity) x.
    """
    cap = tf.cast(state, x.dtype)
    return tf.clip_by_value(x, -cap, cap)

  def GetEndRange(self):
    """Returns the range of values that are clipped towards the end of training.

    This is always a constant and is used by downstream systems.

    Returns:
      Tuple of (min, max).
    """
    return (-self.params.end_cap, self.params.end_cap)

  def _Value(self):
    """Returns the current clipping cap."""
    p = self.params
    start_step = tf.cast(p.start_step, tf.float32)
    end_step = tf.cast(p.end_step, tf.float32)
    current_step = tf.cast(py_utils.GetGlobalStep(), tf.float32)
    steps_ratio = (
        tf.minimum(end_step - start_step, current_step - start_step)/
        (end_step - start_step))
    rmax_tensor = (
        steps_ratio * p.end_cap + (1.0 - steps_ratio) * p.start_cap)
    return tf.cond(
        tf.less(current_step,
                p.start_step), lambda: tf.cast(p.start_cap, tf.float32),
        lambda: tf.cast(rmax_tensor, tf.float32))


class FakeQuantizationSchedule(BaseClippingCapSchedule):
  """Manages application of fake quantization via a schedule.

  This implementation is a general-purpose clipping cap schedule but also
  works with the Fake Quantization approach used by mobile inference engines.
  It is tightly coupled to the FakeQuantizedLSTMCell. See more exhaustive
  documentation and links there.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'FQSchedule'
    p.Define('clip_start_step', 0,
             'We start gradually narrowing clipping cap from start_step.')
    p.Define('clip_end_step', 15000, 'We reach end_cap by end_step.')
    p.Define('quant_start_step', 15000,
             'Step at which we begin to apply quantization.')
    p.Define('start_cap', 8.0, 'Default clipping/quant start cap.')
    p.Define('end_cap', 1.0, 'Default clipping/quant end cap.')
    p.Define('bits', 8, 'Default quantized bit depth.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    # We may relax this constraint at some point to allow gradual quantization
    # but enforce for now as it is easy to mess up and we have not evaluated
    # how it would work otherwise.
    assert p.quant_start_step >= p.clip_end_step, (
        'quant_start_step must be >= clip_end_step')

  @property
  def is_quantized(self):
    return True

  @property
  def bits(self):
    p = self.params
    return p.bits

  def GetEndRange(self):
    """Public method to get the final range as a constant.

    Note that this returns the "ideal" end range (i.e. -1..1) as opposed to
    the actual range, which has its upper bound slightly adjusted based on
    the bit depth of the quantized type. In this sense, this value is a lie,
    but it is a consistent lie that can be corrected for downstream by the
    inference engine once it has inferred the actual quantized types being
    used.

    Note that this also assumes the default start/end caps. Some internal
    parts may use altered caps or bit depths.

    Returns:
      Tuple of (min, max) for the final range.
    """
    p = self.params
    return (-p.end_cap, p.end_cap)

  def GetQuantizedEndRange(self, end_cap=None, bits=None):
    """Gets the quantized ending range.

    Unlike GetEndRange(), this takes quantization effects into account.

    Args:
      end_cap: Override end_cap value.
      bits: Override bits value.
    Returns:
      Tuple of (min, max) for the final range.
    """
    p = self.params
    if end_cap is None:
      end_cap = p.end_cap
    if bits is None:
      bits = p.bits
    return self._GetQuantizedRangeForCap(end_cap, bits)

  def ApplyConstantClip(self, x, min_value, max_value):
    return tf.quantization.fake_quant_with_min_max_vars(
        x, min_value, max_value, num_bits=self.params.bits)

  def GetState(self, theta):
    """Gets the state from theta."""
    p = self.params
    if p.is_inference:
      # State is not used for inference. Just return dummy.
      return tf.zeros([1], tf.float32)
    else:
      # Calculations/vars need to be float but these can be ints in the params.
      clip_end_step = tf.cast(p.clip_end_step, tf.float32)
      clip_start_step = tf.cast(p.clip_start_step, tf.float32)
      quant_start_step = tf.cast(p.quant_start_step, tf.float32)
      global_step = tf.cast(py_utils.GetGlobalStep(), tf.float32)

      # Will be negative if before clipping starts.
      clip_ratio = (
          tf.minimum(clip_end_step - clip_start_step,
                     global_step - clip_start_step) /
          tf.maximum(1.0, clip_end_step - clip_start_step))
      # Currently fq is either on (1.0) or off (-1.0). Progressive quantization
      # may later occupy 0..1.0.
      fq_ratio = tf.where(global_step < quant_start_step, -1.0, 1.0)

      return tf.stack([clip_ratio, fq_ratio])

  def _GetQuantizedRangeForCap(self, current_cap, bits):
    """Gets the range for the given cap and number of bits.

    Args:
      current_cap: Cap to compute against.
      bits: Number of bits (8, 16, etc).
    Returns:
      If current_cap is a python float, the result will be a float. If a Tensor
      scalar, then a Tensor scalar.
    """
    dt_max = 2**(bits - 1)  # i.e. 8bit = 128, 16bit = 32768
    return -current_cap, current_cap * (dt_max - 1) / dt_max

  def _GetCurrentMinMax(self,
                        state,
                        start_cap,
                        end_cap,
                        bits,
                        fixate_to_end_state=False):
    """Gets the current min/max for the bit depth and caps.

    Args:
      state: Clipping state.
      start_cap: Starting cap.
      end_cap: Ending cap once clipping saturates.
      bits: Number of bits of the quantized datatype.
      fixate_to_end_state: Whether to fixate the cap to the end state.
    Returns:
      (min_value, max_value) as python scalars or 0D Tensors (
          if not fixate_to_end_state).
    """
    if fixate_to_end_state:
      current_cap = end_cap
    else:
      clip_ratio = state[0] if not fixate_to_end_state else 1.0
      current_cap = clip_ratio * end_cap + (1.0 - clip_ratio) * start_cap
    return self._GetQuantizedRangeForCap(current_cap, bits)

  def ApplyClippingWithState(self,
                             state,
                             x,
                             start_cap=None,
                             end_cap=None,
                             bits=None):
    """Applies clipping.

    The start_cap, end_cap and bits can be set explicitly and take the default
    if None.

    Args:
      state: Clipping state.
      x: Tensor to clip.
      start_cap: Clipping value at the start of the ramp.
      end_cap: Clipping value at the end of the ramp.
      bits: Number of bits to quantize to.
    Returns:
      x with clipping applied.
    """
    p = self.params
    if start_cap is None:
      start_cap = p.start_cap
    if end_cap is None:
      end_cap = p.end_cap
    if bits is None:
      bits = p.bits
    if p.is_inference:
      # For inference, we assume that both clipping and quantization have
      # saturated and just output a saturated quant op.
      min_value, max_value = self._GetCurrentMinMax(
          state, start_cap, end_cap, bits, fixate_to_end_state=True)
      # Note that the inference version uses the *_args variant, which requires
      # constants for min/max. The _GetCurrentMinMax will return (python)
      # constants if fixating. This is fragile but works around a Toco bug
      # if trying to run on the *_vars form because it can't seem to read
      # 0D tensors. This form has the benefit of blowing up at export time
      # if the min/max aren't constant.
      return _CopyShape(
          x,
          tf.quantization.fake_quant_with_min_max_args(
              x, min_value, max_value, num_bits=bits))

    # Non-inference.
    def Clipped():
      clip_ratio = state[0]
      min_value, max_value = self._GetCurrentMinMax(state, start_cap, end_cap,
                                                    bits)
      min_value = tf.stop_gradient(min_value)
      max_value = tf.stop_gradient(max_value)
      return tf.where(clip_ratio >= 0.0,
                      (lambda: tf.clip_by_value(x, min_value, max_value))(),
                      (lambda: x)())

    def Quantized():
      min_value, max_value = self._GetCurrentMinMax(state, start_cap, end_cap,
                                                    bits)
      min_value = tf.stop_gradient(min_value)
      max_value = tf.stop_gradient(max_value)
      return tf.quantization.fake_quant_with_min_max_vars(
          x, min_value, max_value, num_bits=bits)

    # Quantization will implicitly clip, so if we are in the quant phase, just
    # do that. Otherwise, clip (which will return identity if not in that
    # phase yet).
    fq_ratio = state[1]
    # return _CopyShape(x, Clipped())
    return _CopyShape(x, tf.where(fq_ratio <= 0.0, Clipped(), Quantized()))


class SymmetricScheduledClipQDomain(FakeQDomain):
  """A quantization domain that does symmetric scheduled clipping.

  This contains a BaseClippingCapSchedule which handles the actual clipping. It
  defaults to a FakeQuantizationSchedule.

  This clipping domain will aid in quantizing layers that are known to tolerate
  operation within known ranges (such as LSTM cells). The clipping range will
  converge over a range of steps and is setup to match ideal, symmetric ranges
  for quantized types.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('cc_schedule', FakeQuantizationSchedule.Params(),
             'Quantization clipping schedule.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    self.CreateChild('cc_schedule', p.cc_schedule)

  @property
  def bits(self):
    return self.cc_schedule.bits

  def QuantizeWeight(self, w):
    return self.cc_schedule.ApplyClipping(self.theta.cc_schedule, w)

  def QuantizeNaturalRange(self, t, min_value, max_value):
    # Note: We apply the scheduled clip here, completely overriding the
    # known natural range. This is intentional and assumes that when this
    # layer is used for symmetric clipping, it is applied uniformly to all
    # active elements.
    return self.cc_schedule.ApplyClipping(self.theta.cc_schedule, t)

  def QuantizeConstantRange(self, t, min_value, max_value):
    # Constant ranges, such as padding are handled separately. They are merely
    # constrained to the given range and assumed to be quantizable as-is.
    # This is used for padding.
    return tf.clip_by_value(t, min_value, max_value)

  def QuantizeAct(self, act_name, act, eval_only=False):
    if eval_only and not self.do_eval:
      return act
    else:
      return self.cc_schedule.ApplyClipping(self.theta.cc_schedule, act)


class _CountedMinMaxAccumulator(base_layer.Accumulator):
  """Accumulator for a counted min/max.

  Represented as a tensor of shape [count, min, max]. Every update
  increases the count and expands the min/max (initially zeros).
  """

  def __init__(self, dtype):
    super().__init__()
    self.dtype = dtype

  def DefaultValue(self):
    return tf.zeros([3], dtype=self.dtype, name='qstate_zero')

  def Update(self, new_value):
    state0 = self.GetValue()
    state1 = tf.stack([
        state0[0] + new_value[0],
        tf.minimum(state0[1], new_value[1]),
        tf.maximum(state0[2], new_value[2]),
    ])
    self.SetValue(state1)


class PassiveAsymQDomain(FakeQDomain):
  """A quantization domain that does passive, asymmetric quantization.

  See: https://arxiv.org/abs/1712.05877

  This quantization domain will adjust to min/max ranges during training
  time, recording them into vars via an exponential moving average and then
  applying them at eval/inference time.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('bits', 8, 'Default quantized bit depth.')
    p.Define('ema_decay', 0.99, 'Moving average decay.')
    p.Define('default_min', -1.0,
             'Default minimum value (so initial graphs are valid).')
    p.Define('default_max', 1.0,
             'Default maximum value (so initial graphs are valid).')
    p.Define('quantize_weight_epsilon', 0.0,
             'Default epsilon for weight quantization to prevent zero range.')
    p.Define(
        'delay_start_steps', 0,
        'Delays applying quantization at training time until after '
        'this many steps. 0 = start immediately. -1 = start never. '
        'This is often needed to allow the model to reach some level '
        'of convergence prior to applying quantization. Only affects '
        'training (not eval/inference).')
    p.Define('freeze', False, 'Freeze quantization parameters')
    return p

  def __init__(self, params):
    super().__init__(params)
    self._qvars = py_utils.NestedMap()  # var_name -> tf.Variable

  def _CreateLayerVariables(self):
    # Save a scope for lazily created variables.
    with tf.variable_scope('q'):
      self._qvars_scope = tf.get_variable_scope()

  def _MaybeFakeQuant(self, inputs, min_v, max_v, num_bits):
    p = self.params

    def Apply():
      return tf.quantization.fake_quant_with_min_max_vars(
          inputs, min_v, max_v, num_bits=num_bits)

    if p.delay_start_steps != 0 and not self.do_eval:
      if p.delay_start_steps == -1:
        return inputs
      return tf.where(py_utils.GetGlobalStep() >= p.delay_start_steps, Apply(),
                      inputs)
    else:
      return Apply()

  @property
  def bits(self):
    p = self.params
    return p.bits

  def QuantizeWeight(self, w):
    p = self.params
    w_min = tf.reduce_min(w)
    w_max = tf.reduce_max(w)
    # NOTE: We force a small, non-zero range because otherwise, zero weights
    # can cause downstream inference engines to blow up.
    w_min = tf.minimum(w_min, -p.quantize_weight_epsilon)
    w_max = tf.maximum(w_max, p.quantize_weight_epsilon)
    quant_w = self._MaybeFakeQuant(w, w_min, w_max, num_bits=p.bits)
    if self.do_eval:
      return quant_w
    else:
      # If quantizing during training, skip quantization if it produces
      # NANs. Sometimes early in the training process, things are unstable
      # and ranges can produce numerical instability that makes it
      # impossible to perform a fake_quant.
      quant_w_has_nans = tf.math.is_nan(quant_w)
      return tf.where(quant_w_has_nans, w, quant_w)

  def QuantizeNaturalRange(self, t, min_value, max_value):
    p = self.params
    return self._MaybeFakeQuant(t, min_value, max_value, num_bits=p.bits)

  def QuantizeConstantRange(self, t, min_value, max_value):
    p = self.params
    return self._MaybeFakeQuant(t, min_value, max_value, num_bits=p.bits)

  def TrackQActs(self,
                 *act_names: str,
                 shape: Optional[Iterable[int]] = None,
                 feature_axes: Optional[Iterable[int]] = None):
    super().TrackQActs(*act_names, shape=shape, feature_axes=feature_axes)
    p = self.params
    for act_name in act_names:
      # Create accumulator
      accumulator_name = self._GetAccumulatorNameForTensor(act_name)
      self.RegisterAccumulator(accumulator_name,
                               _CountedMinMaxAccumulator(p.dtype))
      # Register vars.
      min_pc = py_utils.WeightParams(
          (), py_utils.WeightInit.Constant(p.default_min), p.dtype)
      max_pc = py_utils.WeightParams(
          (), py_utils.WeightInit.Constant(p.default_max), p.dtype)
      self._CreateQStateVar(act_name, 'min', min_pc)
      self._CreateQStateVar(act_name, 'max', max_pc)

  def QuantizeAct(self, act_name, act, eval_only=False):
    p = self.params
    # Always straddle a real zero point.
    if self.do_eval:
      # At eval/inference time, use the memorized range.
      # Important: Don't capture these variables in training mode so as to
      # avoid extra/unnecessary captures.
      min_var = self._GetQStateVar(act_name, 'min')
      max_var = self._GetQStateVar(act_name, 'max')
      return self._MaybeFakeQuant(act, min_var, max_var, num_bits=p.bits)
    else:
      # At training time, use the batch calculated min/max.
      accumulator_name = self._GetAccumulatorNameForTensor(act_name)
      # Calculate min/max for this batch.
      batch_min = tf.reduce_min(act)
      batch_max = tf.reduce_max(act)
      # NOTE: This QDomain was implemented such that batch_min <= 0.0 and
      # batch_max >= 0.0 even if that is not the case in the input data.
      batch_min = tf.minimum(batch_min, 0.0)
      batch_max = tf.maximum(batch_max, 0.0)

      # New state.
      state1 = tf.stack([1.0, batch_min, batch_max])
      self.accumulators[accumulator_name].Update(state1)

      # Results.
      if eval_only:
        # If only quantizing at eval time, still record ranges as above
        # but don't quantize.
        quant_act = act
      else:
        # If quantizing during training, skip quantization if it produces
        # NANs. Sometimes early in the training process, things are unstable
        # and ranges can produce numerical instability that makes it
        # impossible to perform a fake_quant.
        quant_act = self._MaybeFakeQuant(
            act, batch_min, batch_max, num_bits=p.bits)
        # TODO(laurenzo): Plumb quant_act_has_nans through state and report.
        quant_act_has_nans = tf.math.is_nan(quant_act)
        quant_act = tf.where(quant_act_has_nans, act, quant_act)
      summary_utils.histogram(f'{self._qvars_scope.name}/{act_name}', act)
      return quant_act

  def PostTrainingStepUpdate(self):
    if self.params.freeze:
      return super().PostTrainingStepUpdate()
    ops = [super().PostTrainingStepUpdate()]
    for act_name in self.act_names:
      ops.extend(self._RecordTensor(act_name))
      self._SummarizeTensor(act_name)
    return tf.group(ops)

  def _GetAccumulatorNameForTensor(self, act_name):
    return f'qact_{act_name}'

  def _GetQStateVarName(self, act_name, suffix):
    return f'{act_name}_{suffix}'

  def _CreateQStateVar(self, act_name, suffix, params):
    name = self._GetQStateVarName(act_name, suffix)
    assert name not in self._qvars, 'QState var already exists: %s' % name
    var_name = f'{self._qvars_scope.name}/{name}'
    with tf.variable_scope(py_utils.GetGlobalVariableScope()):
      v = py_utils.CreateVariable(var_name, params, trainable=False)
    self._qvars[name] = v
    return v

  def _GetQStateVar(self, act_name, suffix):
    v = self._qvars[self._GetQStateVarName(act_name, suffix)]
    return v

  def _SummarizeTensor(self, act_name):
    min_var = self._GetQStateVar(act_name, 'min')
    max_var = self._GetQStateVar(act_name, 'max')
    # foo/q/somet_min:0 -> foo/q/somet_min
    summary_name_min = min_var.name.split(':')[0]
    summary_name_max = max_var.name.split(':')[0]
    summary_utils.scalar(summary_name_min, min_var)
    summary_utils.scalar(summary_name_max, max_var)

  def _RecordTensor(self, act_name):
    p = self.params
    if self.do_eval:
      return []

    accumulator_name = self._GetAccumulatorNameForTensor(act_name)
    accumulator = self.accumulators[accumulator_name]
    min_var = self._GetQStateVar(act_name, 'min')
    max_var = self._GetQStateVar(act_name, 'max')

    # Unpack state tensor.
    current_value = accumulator.GetValue()
    count = current_value[0]
    min_value = current_value[1]
    max_value = current_value[2]
    accumulator.Reset()

    def Ema(variable, value):
      return (1.0 - p.ema_decay) * (variable - value)

    # Note that small floating point issues can cause ranges that naturally
    # begin or end at zero to move slightly past, causing hard failures
    # downstream (checks that all ranges straddle zero). We therefore repeat
    # the straddling constraint here.
    return [
        tf.assign(
            min_var,
            tf.minimum(
                0.,
                min_var - tf.where(count > 0., Ema(min_var, min_value), 0.))),
        tf.assign(
            max_var,
            tf.maximum(
                0.,
                max_var - tf.where(count > 0., Ema(max_var, max_value), 0.))),
    ]


def _CopyShape(from_t, to_t):
  """Sets the shape of from_t to to_t."""
  if isinstance(from_t, tf.Tensor) and isinstance(to_t, tf.Tensor):
    to_t.set_shape(from_t.shape)
  return to_t
