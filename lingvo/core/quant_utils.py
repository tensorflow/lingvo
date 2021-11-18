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
"""Utilities for model quantization."""

import enum

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

  - QWeight: Tags a tensor (typically a var) as a weight quantized type.
  - QRAct(act, dist: QDistribution): Tags a tensor as an activation
    with a known or configurable quantization range.
  - QTensor: Tags a tensor as a generic quantized intermediate value.
    These are also tagged with a layer-unique name. All QTensors with the
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

    - qout_name: Name of QTensor (setup with TrackQTensor) for dynamic range
      tracking.
    - qmin/qmax/qdomain: Constant min/max range plus optional QDomain name to
      resolve against. Typically, only qmin/qmax are used.

  Functions that have a natural output range will have default values for
  qmin/qmax so that they just work. Functions that do not have a natural
  output range must have either qout_name or qmin/qmax specified manually.

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
  - qmatmul (defers to `.py_utils.Matmul` and only accepts rank-2 tensors)
  - qbatchmatmul (defers to `tf.matmul` directly)
  - qconv1d
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

    self._tracked_tensors = dict()  # tracked t_name -> (QDomain)
    self._aqt_weights = dict()  # aqt w_name -> (Qdomain)
    self._qstate = None  # t_name -> Tensor

    # Instantiate quantization domains.
    self._qdomains = dict()  # Dict of qdname -> QDomain or None
    for qdname in dir(p.qdomain):
      qdparams = p.qdomain.Get(qdname)
      if qdparams is None:
        continue
      assert issubclass(
          qdparams.cls,
          QDomain), ('Expected quantized domain %s to extend QDomain' % qdname)
      qdchild_name = 'qdomain_' + qdname
      self.CreateChild(qdchild_name, qdparams)
      self._qdomains[qdname] = self.children[qdchild_name]
    self._AddQuantizationFunctions()

  def _CreateChildrenVariables(self):
    # Backwards compatibility: child.InstantiateVariables() in custom scope.
    p = self.params
    with tf.variable_scope(p.name + '/q'):
      for qdomain in self._qdomains.values():
        qdomain.InstantiateVariables()
    super()._CreateChildrenVariables()

  def TrackQTensor(self, *t_names, domain='default'):
    r"""Creates one or more QTensors for later use.

    Any tensor that will later be quantized must be created first, preferably
    in _CreateLayerVariables().

    Along with a list of tensor names to create, they can be associated with
    a 'domain'. Most layers are simple enough to only have a single quantization
    domain (QDomain), typically 'default'. However, additional QDomains can
    be defined as parameters to control fine grained aspects of quantization.

    Args:
      *t_names: Positional parameters are taken to be QTensor names to create.
      domain: name of the qdomain to use
    """
    qd = self._GetQDomain(domain)
    for t_name in t_names:
      self._tracked_tensors[t_name] = qd
      if qd:
        qd.CreateTensor(t_name)

  def CreateAqtWeight(self,
                      w_name,
                      shape,
                      feature_axis,
                      domain='weight',
                      *,
                      legacy_aqt_w_name=None):
    """Creates Quantized weights for later use.

    Weight that will later be quantized must be created first, preferably
    in _CreateLayerVariables().

    Args:
      w_name: Positional parameters are taken to be QTensor names to create.
      shape: Shape of the weight.
      feature_axis: axis corresponding to output channel/feature for weights.
      domain: Custom domain to match (defaults to 'weight').
      legacy_aqt_w_name: Used for compatibility with old checkpoints.
    """
    qd = self._GetQDomain(domain)
    self._aqt_weights[w_name] = qd
    if qd:
      qd.CreateTensorWithShape(w_name, shape, feature_axis, legacy_aqt_w_name)

  def QTensor(self, t_name, t, eval_only=False):
    """Quantizes a general tensor input/output in one step.

    t_name must have been previously created via TrackQTensor.

    Args:
      t_name: Previously created QTensor t_name to quantize to.
      t: Tensor to quantize.
      eval_only: Whether to only apply quantization pressure at eval time.

    Returns:
      The tensor, quantized.
    """
    assert t_name in self._tracked_tensors, (
        ('Call to QTensor without first calling TrackQTensor: %s '
         '(all known = %r)') % (t_name, list(self._tracked_tensors.keys())))
    qd = self._tracked_tensors[t_name]
    if not qd:
      return t
    else:
      return qd.QuantizeTensors(t_name, [t], eval_only=eval_only)[0]

  def QWeight(self, w, domain='weight'):
    """Quantizes a weight.

    Args:
      w: The weight tensor.
      domain: Custom domain to match (defaults to 'weight' or 'default').
    Returns:
      The weights quantized.
    """
    qd = self._GetQDomain(domain)
    return qd.QuantizeWeight(w) if qd else w

  def ToAqtWeight(self, w_name, w, feature_axis, expected_scale_shape=None):
    """Quantized integer weight AQT style.

    This only scales, rounds and clips; resulting quantized weight would be
    either integer or integer emulated in float.

    w_name must have been previously created via CreateAqtWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      w: The weight tensor.
      feature_axis: axis corresponding to output channel/feature for weights.
      expected_scale_shape: Optional shape to verify if scale shape is expected.
        Defaults to None.

    Returns:
      Quantized weights.
    """
    assert w_name in self._aqt_weights, (
        ('Call to ToAqtWeight without first calling CreateAqtWeight: %s '
         '(all known = %r)') % (w_name, list(self._aqt_weights.keys())))

    # If w is a variable on this layer, then validate that w_name matches the
    # name given to w in self.CreateVariable.
    if w.ref() in self.vars:
      expected_name = w.name.split('/')[-2]  # model/path/layer/weight/var:0
      if w_name != expected_name:
        raise ValueError(
            f'Expected the AQT weight name to match the name of non-AQT '
            f"weight, but got '{w_name}' and '{expected_name}'."
            f'\nFull weight info:\n  {w}')

    qd = self._aqt_weights[w_name]
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

    w_name must have been previously created via CreateAqtWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      out: The tensor to rescale.
      merge_feature_axes: whether or the feature axes have been reshaped into a
        single axis in 'out'.

    Returns:
      Rescaled output.
    """
    assert w_name in self._aqt_weights, (
        ('Call to FromAqtWeight without first calling CreateAqtWeight: %s '
         '(all known = %r)') % (w_name, list(self._aqt_weights.keys())))
    qd = self._aqt_weights[w_name]
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
    assert w_name in self._aqt_weights, (
        ('Call to ToAqtConv without first calling CreateAqtWeight: %s '
         '(all known = %r)') % (w_name, list(self._aqt_weights.keys())))
    qd = self._aqt_weights[w_name]
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
    assert w_name in self._aqt_weights, (
        ('Call to FromAqtConv without first calling CreateAqtWeight: %s '
         '(all known = %r)') % (w_name, list(self._aqt_weights.keys())))
    qd = self._aqt_weights[w_name]
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

    w_name must have been previously created via CreateAqtWeight.

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
    assert w_name in self._aqt_weights, (
        ('Call to ToAqtInputs without first calling CreateAqtWeight: %s '
         '(all known = %r)') % (w_name, list(self._aqt_weights.keys())))
    qd = self._aqt_weights[w_name]
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

    w_name must have been previously created via CreateAqtWeight.

    Args:
      w_name: Previously created w_name QWeight to quantize weight.
      output: The tensor to rescale.

    Returns:
      Rescaled output.
    """
    assert w_name in self._aqt_weights, (
        ('Call to FromAqtWeight without first calling CreateAqtWeight: %s '
         '(all known = %r)') % (w_name, list(self._aqt_weights.keys())))
    qd = self._aqt_weights[w_name]
    return qd.FromAqtMatmul(w_name, output) if qd else output

  def ToAqtActActInputs(self,
                        act_lhs,
                        act_rhs,
                        *,
                        act_lhs_distribution=QDistribution.SYMMETRIC,
                        act_rhs_distribution=QDistribution.SYMMETRIC,
                        domain=None):
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

  def FromAqtActActMatmul(self, output, domain=None):
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

  def _GetQDomain(self, domain):
    """Gets the QDomain matching a given domain name.

    Args:
      domain: User specified domain name.

    Returns:
      The requested QDomain, the 'default' QDomain or None.
    """
    qd = self._qdomains.get(domain)
    if qd:
      return qd
    qd = self._qdomains.get('default')
    return qd

  def GetQDomainParams(self, domain):
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
                  qmin=None,
                  qmax=None,
                  qdomain=None,
                  **op_kwargs):
        """Wraps a native op."""
        if (qout_name is None and (qmin is None or qmax is None) and
            dist is None):
          raise ValueError(
              f'Quantized op "{op_name}" requires qout_name (QTensor name) or '
              'qmin/qmax to be set.')

        # Provide a better default name if none provided.
        if 'name' not in op_kwargs and qout_name is not None:
          op_kwargs['name'] = '%s_%s' % (op_name, qout_name)

        # Invoke original.
        y = op(*op_args, **op_kwargs)

        # Handle the output.
        if qout_name is not None:
          y = self.QTensor(qout_name, y)
        elif qmin is not None:
          qd = self._GetQDomain(qdomain)
          if qd:
            y = qd.QuantizeNaturalRange(y, qmin, qmax)
        else:
          y = self.QRAct(y, dist, qdomain)
        return y

      self.AddFunction(op_name, Wrapped)

    # Supported quantized functions.
    WrapOp('qadd', tf.add)
    WrapOp('qsubtract', tf.subtract)
    WrapOp('qmultiply', tf.multiply)
    WrapOp('qmatmul', py_utils.Matmul)
    WrapOp('qbatchmatmul', tf.matmul)
    WrapOp('qconv1d', tf.nn.conv1d)
    WrapOp('qtanh', tf.tanh, dist=QDistribution.TANH)
    WrapOp('qsigmoid', tf.sigmoid, dist=QDistribution.SIGMOID)
    WrapOp('qsoftmax', tf.nn.softmax, dist=QDistribution.SOFTMAX)
    WrapOp('qlog', tf.math.log)
    WrapOp('qlogsoftmax', tf.nn.log_softmax, dist=QDistribution.LOG_SOFTMAX)
    WrapOp('qrelu', tf.nn.relu, dist=QDistribution.RELU)
    WrapOp('qrelu6', tf.nn.relu6, dist=QDistribution.RELU6)
    WrapOp(
        'qrandom_uniform', tf.random.uniform, dist=QDistribution.RANDOM_UNIFORM)


class QDomain(base_layer.BaseLayer):
  """Base class for a quantization domain layer.

  This implementation doubles as a no-op quantization domain.
  """

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

  def CreateTensor(self, t_name):
    """Creates a QTensor with t_name.

    Args:
      t_name: Unique name (within layer) for this tensor.
    """
    pass

  def CreateTensorWithShape(self,
                            t_name,
                            shape,
                            feature_axis,
                            legacy_aqt_t_name=None):
    """Creates a QTensor with t_name and given shape.

    Args:
      t_name: Unique name (within layer) for this tensor.
      shape: Expected shape of the tensor.
      feature_axis: axis corresponding to output channel/feature for weights.
      legacy_aqt_t_name: Used for compatibility with old checkpoints.
    """
    pass

  def QuantizeTensors(self, t_name, ts, eval_only=False):
    """Quantizes a tensor with t_name previously created with CreateTensor.

    If applicable, each of the passed tensors contributes to a shared
    range.

    Args:
      t_name: Tensor name.
      ts: List of tensors to quantize.
      eval_only: Whether to only apply quantization pressure at eval time.

    Returns:
      Quantized tensors.
    """
    return ts


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

  def QuantizeTensors(self, t_name, ts, eval_only=False):
    if eval_only and not self.do_eval:
      return ts
    else:
      return [
          self.cc_schedule.ApplyClipping(self.theta.cc_schedule, t) for t in ts
      ]


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
    return p

  def __init__(self, params):
    super().__init__(params)

    self._t_names = set()  # set of known t_name (from CreateTensor)
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

  def CreateTensor(self, t_name):
    p = self.params
    assert t_name not in self._t_names, (
        'QTensor already registered: %s' % t_name)
    self._t_names.add(t_name)

    # Create accumulator
    accumulator_name = self._GetAccumulatorNameForTensor(t_name)
    self.RegisterAccumulator(accumulator_name,
                             _CountedMinMaxAccumulator(p.dtype))
    # Register vars.
    min_pc = py_utils.WeightParams((),
                                   py_utils.WeightInit.Constant(p.default_min),
                                   p.dtype)
    max_pc = py_utils.WeightParams((),
                                   py_utils.WeightInit.Constant(p.default_max),
                                   p.dtype)
    self._CreateQStateVar(t_name, 'min', min_pc)
    self._CreateQStateVar(t_name, 'max', max_pc)

  def QuantizeTensors(self, t_name, ts, eval_only=False):
    p = self.params
    # Always straddle a real zero point.
    if self.do_eval:
      # At eval/inference time, use the memorized range.
      # Important: Don't capture these variables in training mode so as to
      # avoid extra/unnecessary captures.
      min_var = self._GetQStateVar(t_name, 'min')
      max_var = self._GetQStateVar(t_name, 'max')
      return [
          self._MaybeFakeQuant(t, min_var, max_var, num_bits=p.bits) for t in ts
      ]
    else:
      # At training time, use the batch calculated min/max.
      accumulator_name = self._GetAccumulatorNameForTensor(t_name)
      # Calculate min/max for all tensors.
      batch_min = 0.0
      batch_max = 0.0
      for t in ts:
        batch_min = tf.minimum(tf.reduce_min(t), batch_min)
        batch_max = tf.maximum(tf.reduce_max(t), batch_max)

      # New state.
      state1 = tf.stack([1.0, batch_min, batch_max])
      self.accumulators[accumulator_name].Update(state1)

      # Results.
      ts_out = []
      for i, t in enumerate(ts):
        if eval_only:
          # If only quantizing at eval time, still record ranges as above
          # but don't quantize.
          quant_t = t
        else:
          # If quantizing during training, skip quantization if it produces
          # NANs. Sometimes early in the training process, things are unstable
          # and ranges can produce numerical instability that makes it
          # impossible to perform a fake_quant.
          quant_t = self._MaybeFakeQuant(
              t, batch_min, batch_max, num_bits=p.bits)
          # TODO(laurenzo): Plumb quant_t_has_nans through state and report.
          quant_t_has_nans = tf.math.is_nan(quant_t)
          quant_t = tf.where(quant_t_has_nans, t, quant_t)
        ts_out.append(quant_t)
        summary_utils.histogram(
            '%s/%s_%d' % (self._qvars_scope.name, t_name, i), t)
      return ts_out

  def PostTrainingStepUpdate(self):
    ops = [super().PostTrainingStepUpdate()]
    for t_name in self._t_names:
      ops.extend(self._RecordTensor(t_name))
      self._SummarizeTensor(t_name)
    return tf.group(ops)

  def _CreateQStateVar(self, t_name, suffix, params):
    name = t_name + '_' + suffix
    assert name not in self._qvars, 'QState var already exists: %s' % name
    var_name = self._qvars_scope.name + '/' + name
    with tf.variable_scope(py_utils.GetGlobalVariableScope()):
      v = py_utils.CreateVariable(var_name, params, trainable=False)
    self._qvars[name] = v
    return v

  def _GetAccumulatorNameForTensor(self, t_name):
    return 'qtensor_' + t_name

  def _GetQStateVar(self, t_name, suffix):
    v = self._qvars[t_name + '_' + suffix]
    return v

  def _SummarizeTensor(self, t_name):
    min_var = self._GetQStateVar(t_name, 'min')
    max_var = self._GetQStateVar(t_name, 'max')
    # foo/q/somet_min:0 -> foo/q/somet_min
    summary_name_min = min_var.name.split(':')[0]
    summary_name_max = max_var.name.split(':')[0]
    summary_utils.scalar(summary_name_min, min_var)
    summary_utils.scalar(summary_name_max, max_var)

  def _RecordTensor(self, t_name):
    p = self.params
    if self.do_eval:
      return []

    accumulator_name = self._GetAccumulatorNameForTensor(t_name)
    accumulator = self.accumulators[accumulator_name]
    min_var = self._GetQStateVar(t_name, 'min')
    max_var = self._GetQStateVar(t_name, 'max')

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
