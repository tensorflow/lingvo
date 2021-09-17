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
"""Optimizers."""

import copy
import re

import lingvo.compat as tf
from lingvo.core import adagraft
from lingvo.core import base_layer
from lingvo.core import distributed_shampoo
from lingvo.core import gshard_utils
from lingvo.core import py_utils
from lingvo.core import summary_utils


class Base(base_layer.BaseLayer):
  """Base class for all optimizers."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = cls.__name__
    p.Define(
        'use_bf16_gradients_ar', False,
        'Whether to use bfloat16 dtype for gradients all-reduce. '
        'This applies to TPU only.')
    p.Define('add_summary_in_apply', True, 'Whether to add summary in Apply.')
    return p

  def __init__(self, params):
    super().__init__(params)

    self._supports_eager = False
    # The cached optimizer.
    self._optimizer = None

  def GetOptimizer(self, lr):
    """Returns the TF optimizer object."""
    raise NotImplementedError('Abstract method')

  def AddSummary(self, lr, optimizer, var_grad):
    """Adds summary if needed."""
    pass

  def ComputeGradients(self, loss, vmap, *args, **kwargs):
    """Allows subclasses control computation of gradients."""
    kwargs['use_bf16_gradients_ar'] = self.params.use_bf16_gradients_ar
    return py_utils.ComputeGradients(loss, vmap, *args, **kwargs)

  def VarReuseForSlotVars(self):
    """Multi-task models require AUTO_REUSE for var sharing."""
    var_reuse = False
    if py_utils.GetOpportunisticVariableReuse():
      var_reuse = tf.AUTO_REUSE
    return var_reuse

  def Apply(self, lr, var_grad):
    """Applies the gradient to the variable.

    Args:
      lr: A scalar. The base learning rate.
      var_grad: A `.NestedMap` of (var, grad) pairs.

    Returns:
      The variable update op.
    """
    if py_utils.IsEagerMode() and not self._supports_eager:
      raise ValueError(
          f'{type(self)} does not support eager mode. '
          'Please use a different optimizer or file a bug to add support.')

    if self._optimizer is None:
      self._optimizer = self.GetOptimizer(lr)
    else:
      if py_utils.IsEagerMode():
        # TODO(jiaweix): we need a mechanism for V1 optimizers
        self._optimizer.learning_rate = lr
      else:
        self._optimizer = self.GetOptimizer(lr)

    def _Apply():
      if self.params.use_bf16_gradients_ar:
        return self._optimizer.apply_gradients(
            [(tf.cast(g, tf.float32), v) for (v, g) in var_grad.Flatten()],
            name='meta_backprop')
      else:
        return self._optimizer.apply_gradients(
            [(g, v) for (v, g) in var_grad.Flatten()], name='meta_backprop')

    # Many optimizers, e.g., Adam, Adagrad, etc., create
    # variables. We need to ensure name scope and variable scope are
    # cleared. Otherwise, tpu.batch_parallel does not work.
    with tf.name_scope(None):
      with tf.variable_scope(
          tf.VariableScope(use_resource=True,
                           reuse=self.VarReuseForSlotVars())):
        var_update_op = _Apply()
    if self.params.add_summary_in_apply:
      self.AddSummary(lr, self._optimizer, var_grad)
    return var_update_op

  def ApplyPostTrainingLoop(self):
    """Applies any computation to run after each tpu trainining loop.

    Returns:
      Ops to run after training loop ends.
    """
    return tf.no_op()


class CompositeOptimizer(Base):
  """Composite Optimizer.

  A composite optimizer is composed of one or more Lingvo Optimizer objects
  where regex specifies which variables should use which optimizer. The
  optimizer_map dictionary must specify a default_optimizer regex to a
  (Lingvo Optimizer, learning rate) tuple which will be applied to all variables
  which do not match an earlier regex.

  For example,

  optimizer_map = {'a': Adam, 'b': Adagrad, 'default_optimizer': SGD}

  will apply Adam to all variables which contain an 'a' in their name, apply
  Adagrad to all variables which contain a 'b' in their name, and apply SGD to
  the variables which do not contain either 'a' or 'b'.

  If a non-default_optimizer matches more than one variable -- in this example
  variables with both 'a' and 'b' in their name -- an exception is thrown.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'optimizer_map', None,
        'Mapping of variable regex to (Lingvo Optimizer, learning rate) tuple.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self._optimizer_map = {}
    self._lr_map = {}
    for index, regex in enumerate(params.optimizer_map):
      sub_optimizer, learning_rate = params.optimizer_map[regex]
      self.CreateChild('sub_{}_{}'.format(sub_optimizer.name, index),
                       sub_optimizer)
      self._optimizer_map[regex] = self.children['sub_{}_{}'.format(
          sub_optimizer.name, index)]
      self._lr_map[regex] = learning_rate

    if 'default_optimizer' not in self._optimizer_map:
      raise KeyError('default_optimizer is not found in optimizer_map. Please '
                     'specify a default_optimizer regex and its associated '
                     '(Lingvo Optimizer, learning rate) tuple.')

  def GetOptimizer(self, lr):
    """Returns a dictionary of regex to TF optimizer objects."""
    return {
        k: v.GetOptimizer(self._lr_map[k])
        for k, v in self._optimizer_map.items()
    }

  def Apply(self, lr, var_grad):
    """For each optimizer, apply the gradient to the variable.

    Args:
      lr: A scalar. The base learning rate.
      var_grad: A `.NestedMap` of (var, grad) pairs.

    Returns:
      The variable update op.

    Raises:
      Exception: When the regex overlaps with or does not cover all variables.
    """
    # Override inherited GetOptimizer even though learning rate is unused.
    tf_optimizer_map = self.GetOptimizer(0)
    var_grad_map = {regex: [] for regex in self._optimizer_map}

    for (v, g) in var_grad.Flatten():
      regex_match = 0
      for regex in self._optimizer_map:
        if re.match(regex, v.name):
          var_grad_map[regex].append((g, v))
          regex_match += 1
      if regex_match == 0:
        var_grad_map['default_optimizer'].append((g, v))
      if regex_match > 1:
        raise Exception('Variable {} is matched {} times by regex {}'.format(
            v.name, regex_match, list(self._optimizer_map.keys())))

    def _Apply():
      """Use the matched optimizer to apply the gradients."""
      train_ops = []
      non_default_regex = [
          regex for regex in self._optimizer_map if regex != 'default_optimizer'
      ]
      for regex in self._optimizer_map:
        if var_grad_map[regex]:
          opt = tf_optimizer_map[regex]
          train_ops.append(opt.apply_gradients(var_grad_map[regex]))
          # pylint: disable=cell-var-from-loop, g-long-lambda
          if regex == 'default_optimizer':
            filtered_var_grad = var_grad.FilterKeyVal(lambda k, v: any(
                [re.match(i, v.var.name) for i in non_default_regex]))
          else:
            filtered_var_grad = var_grad.FilterKeyVal(
                lambda k, v: (re.match(regex, v.var.name)))
          # pylint: enable=cell-var-from-loop, g-long-lambda
          self._optimizer_map[regex].AddSummary(self._lr_map[regex], opt,
                                                filtered_var_grad)
      return tf.group(*train_ops, name='composite_optimizer_train_op')

    # Many optimizers, e.g., Adam, Adagrad, etc., create
    # variables. We need to ensure name scope and variable scope are
    # cleared. Otherwise, tpu.batch_parallel does not work.
    var_reuse = False
    if py_utils.GetOpportunisticVariableReuse():
      var_reuse = tf.AUTO_REUSE
    with tf.name_scope(None):
      with tf.variable_scope(
          tf.VariableScope(use_resource=True, reuse=var_reuse)):
        var_update_op = _Apply()
    return var_update_op

  def ApplyPostTrainingLoop(self):
    """Apply computation to run after each tpu training loop for each optimizer.

    Returns:
      Ops to run after training loop ends.
    """
    post_training_ops = [
        opt.ApplyPostTrainingLoop() for _, opt in self._optimizer_map.items()
    ]
    return tf.group(*post_training_ops)


class SGD(Base):
  """SGD."""

  def __init__(self, params):
    super().__init__(params)
    self._supports_eager = True

  def GetOptimizer(self, lr):
    return tf.train.GradientDescentOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('sgd_lr', lr)


class Momentum(Base):
  """Momentum optimizer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'alpha', 0.9, 'The damping factor in the momentum '
        'optimizer. This controls how the velocity (averaged '
        'past gradients) is decayed over time.')
    p.Define('use_nesterov', False, 'True iff use Nesterov')
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.MomentumOptimizer(
        learning_rate=lr, momentum=p.alpha, use_nesterov=p.use_nesterov)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('momentum_lr', lr)


class RMSProp(Base):
  """RMSProp optimizer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decay', 0.9, 'Discounting factor for the history/coming gradient')
    p.Define('momentum', 0.9, 'Momentum in RMSProp.')
    p.Define(
        'epsilon', 1.0,
        'Epsilon term for RMSProp. Small value to avoid zero denominator.')
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.RMSPropOptimizer(
        lr, p.decay, momentum=p.momentum, epsilon=p.epsilon)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('rmsprop_lr', lr)


class Adagrad(Base):
  """Adagrad."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('initial_accumulator_value', 1.0,
             "Adagrad's initial_accumulator_value.")
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdagradOptimizer(
        learning_rate=lr, initial_accumulator_value=p.initial_accumulator_value)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adagrad_lr', lr)
    for v, _ in var_grad.Flatten():
      slot = optimizer.get_slot(v, 'accumulator')
      assert slot is not None
      summary_utils.scalar('optimizer/adagrad_accum_%s' % v.name,
                           tf.reduce_mean(slot))


class AdaDelta(Base):
  """AdaDelta optimizer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('decay', 0.95,
             'Discounting factor for the history/coming gradient')
    p.Define(
        'epsilon', 1e-8,
        'Epsilon term for AdaDelta. Small value to avoid zero denominator.')
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdadeltaOptimizer(
        learning_rate=lr, rho=p.decay, epsilon=p.epsilon)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adadelta_lr', lr)


class Adam(Base):
  """Adam."""

  @classmethod
  def Params(cls):
    if py_utils.IsEagerMode():
      tf.logging.warning('Adam optimizer is not supported in eager mode. '
                         'Automatically converting to AdamV2.')
      return AdamV2.Params()

    p = super().Params()
    p.Define('beta1', 0.9, 'Beta1 for Adam.')
    p.Define('beta2', 0.999, 'Beta2 for Adam.')
    p.Define('epsilon', 1e-6, 'Epsilon for Adam.')
    p.name = 'Adam'
    return p

  @classmethod
  def ParamsA(cls):
    """Convenient method for a commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.997, epsilon=1e-9)

  @classmethod
  def ParamsB(cls):
    """Convenient method for another commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.98, epsilon=1e-9)

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=p.beta1,
        beta2=p.beta2,
        epsilon=p.epsilon,
        name=p.name)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adam_lr', lr)


class AdamV2(Base):
  """Adam from TF2."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('beta1', 0.9, 'Beta1 for Adam.')
    p.Define('beta2', 0.999, 'Beta2 for Adam.')
    p.Define('epsilon', 1e-6, 'Epsilon for Adam.')
    p.name = 'Adam'
    return p

  def __init__(self, params):
    super().__init__(params)
    self._supports_eager = True

  @classmethod
  def ParamsA(cls):
    """Convenient method for a commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.997, epsilon=1e-9)

  @classmethod
  def ParamsB(cls):
    """Convenient method for another commonly used Adam config."""
    return cls.Params().Set(beta1=0.9, beta2=0.98, epsilon=1e-9)

  def GetOptimizer(self, lr):
    p = self.params
    return tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=p.beta1,
        beta_2=p.beta2,
        epsilon=p.epsilon,
        name=p.name)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adam_lr', lr)


class Accumulator(Base):
  """Gradient accumulator wrapper."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('optimizer_tpl', Adam.Params(),
             'Params for the wrapped optimizer.')
    p.Define(
        'accum_steps', 5, 'Number of gradient accumulation steps'
        ' before invoking wrapped optimizer.')
    p.name = 'Accumulator'
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    # Disable tf.summary in control flow ops.
    p.optimizer_tpl.add_summary_in_apply = False
    self.CreateChild('_opt', p.optimizer_tpl)

  def Apply(self, lr, var_grad):
    p = self.params

    def _Acc(vg):
      """Updating accumulators."""

      v, g = vg
      with tf.variable_scope(v.op.name):
        a = py_utils.CreateVariable(
            'grad_accumulator',
            py_utils.WeightParams(v.get_shape(),
                                  py_utils.WeightInit.Constant(0.0),
                                  self.params.dtype),
            trainable=False)
        a = tf.assign_add(a, g)

      return py_utils.VarGrad(v, a)

    var_grad = var_grad.Transform(_Acc)

    def _ApplyAndReset():
      with tf.control_dependencies([
          self._opt.Apply(
              lr, py_utils.ApplyGradMultiplier(var_grad, 1. / p.accum_steps))
      ]):
        return tf.group(
            *[tf.assign(a, tf.zeros_like(a)) for _, a in var_grad.Flatten()])

    if self.params.add_summary_in_apply:
      self.AddSummary(lr, self.GetOptimizer(lr), var_grad)
    return tf.cond(
        tf.equal(
            tf.math.floormod(py_utils.GetGlobalStep(), p.accum_steps),
            p.accum_steps - 1), _ApplyAndReset, lambda: tf.group(tf.no_op()))

  def GetOptimizer(self, lr):
    return self._opt.GetOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    return self._opt.AddSummary(lr, optimizer, var_grad)


class DistributedShampoo(Base):
  """Approximates full-matrix AdaGrad per layer.

  Approximates full-matrix AdaGrad with kronecker-products of two statistics
  matrices based on only the first-order gradients of the layer.

  "Second-order optimization made practical.", 2019
  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer.
  """

  @classmethod
  def Params(cls):
    params = super().Params()
    params.Define('momentum', 0.9, 'Momentum parameter.')
    params.Define('start_preconditioning_steps', 1000,
                  'When to start approximate full matrix preconditioning.')
    params.Define('initial_accumulator_value', 0.0,
                  'Initial accumulator value.')
    params.Define('block_size', 4096, 'Block size for partitioning.')
    params.Define('block_partition_threshold_size', 1000000,
                  'Threshold for block partitioning.')
    params.Define('max_any_dim', 8192,
                  'max dimension before skipping preconditioning altogether.')
    params.Define('matrix_epsilon', 1e-6,
                  'Minimum eigen value used to improve the conditioning.')
    params.Define(
        'second_moment_averaging', 1.0,
        'Averaging coefficient, with special case of (1.0) means sum '
        'of squares while less than 1.0 is RMSProp style moving'
        ' average modification.')
    params.Define(
        'fallback_to_diagonal_dim', 4096,
        'If any dimension is larger than this value, the optimizer falls back'
        ' to the diagonal.')
    params.Define(
        'statistics_computation_frequency', 1,
        'How often to compute statistics. Greater than 1 speeds up training.')
    return params

  def GetOptimizer(self, lr):
    params = self.params
    return distributed_shampoo.DistributedShampoo(
        learning_rate=lr,
        momentum=params.momentum,
        start_preconditioning_steps=params.start_preconditioning_steps,
        initial_accumulator_value=params.initial_accumulator_value,
        matrix_epsilon=params.matrix_epsilon,
        statistics_computation_frequency=(
            params.statistics_computation_frequency),
        second_moment_averaging=params.second_moment_averaging,
        max_any_dim=params.max_any_dim,
        block_size=params.block_size,
        global_step=py_utils.GetGlobalStep())

  def Apply(self, lr, var_grad):
    """Applies the gradient to the variable.

    Args:
      lr: A scalar. The base learning rate.
      var_grad: A `.NestedMap` of (var, grad) pairs.

    Returns:
      The variable update op.
    """
    self._optimizer = self.GetOptimizer(lr)

    def _Apply():
      return self._optimizer.apply_gradients(
          [(g, v) for (v, g) in var_grad.Flatten()], name='meta_backprop')

    # Many optimizers, e.g., Adam, Adagrad, etc., create
    # variables. We need to ensure name scope and variable scope are
    # cleared. Otherwise, tpu.batch_parallel does not work.
    with tf.name_scope(None):
      with tf.variable_scope(
          tf.VariableScope(use_resource=True,
                           reuse=self.VarReuseForSlotVars())):
        var_update_op = _Apply()

    if self.params.add_summary_in_apply:
      self.AddSummary(lr, self._optimizer, var_grad)
    return var_update_op

  def ApplyPostTrainingLoop(self):
    """Applies any computation to run after each tpu trainining loop.

    Returns:
      Ops to run after training loop ends.
    """
    invoke_async_ops = self._optimizer.invoke_async_preconditioner_computation(
        tf.cast(py_utils.GetGlobalStep(), tf.int32))
    assign_ops = self._optimizer.assign_preconditioner_to_host_vars()
    return tf.group(*[invoke_async_ops, assign_ops])

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('distributed_shampoo', lr)


class AdaGraft(Base):
  """Optimizer which combines step size and direction of two optimizers.


  Disentangling Adaptive Gradient Methods from Learning Rates
  Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, Cyril Zhang
  https://arxiv.org/abs/2002.11803
  """

  @classmethod
  def Params(cls):
    params = super().Params()

    params.Define('magnitude_optimizer', None,
                  'Instantiated Optimizer layer providing the step size.')
    params.Define('direction_optimizer', None,
                  'Instantiated Optimizer layer providing the step direction.')
    params.Define(
        'direction_optimizer_lr', None,
        'Custom constant learning rate passed to direction '
        'optimizer. If None, then pass scheduled lr for both.')
    params.Define('use_global_norm', False, 'Whether to graft global l2 norm.')
    params.Define('diagnostic', False, 'Whether to record norm measurements.')

    params.name = 'AdaGraft'
    return params

  def GetOptimizer(self, lr):
    params = self.params

    if params.direction_optimizer_lr is None:
      dir_lr = lr
    else:
      dir_lr = params.direction_optimizer_lr

    magnitude_tf_optimizer = params.magnitude_optimizer.GetOptimizer(lr=lr)
    direction_tf_optimizer = params.direction_optimizer.GetOptimizer(lr=dir_lr)

    return adagraft.AdaGraftOptimizer(
        1.0,
        magnitude_tf_optimizer,
        direction_tf_optimizer,
        use_global_norm=params.use_global_norm,
        diagnostic=params.diagnostic)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adagraft_lr', lr)

    if self.params.diagnostic:  # verbose option
      m_step_norm_total = 0.0
      d_step_norm_total = 0.0

      for v, _ in var_grad.Flatten():  # record layer-wise gradient norms
        m_step_norm = optimizer.get_slot(v, 'm_step_norm')
        d_step_norm = optimizer.get_slot(v, 'd_step_norm')
        summary_utils.scalar('optimizer/m_step_norm_%s' % v.name, m_step_norm)
        summary_utils.scalar('optimizer/d_step_norm_%s' % v.name, d_step_norm)
        m_step_norm_total += m_step_norm**2
        d_step_norm_total += d_step_norm**2

      # record global gradient norms
      m_step_norm_total **= 0.5
      d_step_norm_total **= 0.5
      summary_utils.scalar('optimizer/m_step_norm', m_step_norm_total)
      summary_utils.scalar('optimizer/d_step_norm', d_step_norm_total)
      summary_utils.scalar('optimizer/norm_correction',
                           m_step_norm_total / d_step_norm_total)


def _StepNum():
  return tf.cast(tf.train.get_or_create_global_step(), tf.float32)


def _AdafactorDecayRatePow(exponent, offset=0):
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: a float between 0 and 1
    offset: an optional integer

  Returns:
    a scalar
  """
  return 1.0 - tf.pow((_StepNum() - offset + 1.0), -exponent)


def _AdafactorDecayRateAdam(beta2):
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: a float between 0 and 1

  Returns:
    a scalar
  """
  t = _StepNum() + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
  return decay


# Adaptation of mesh_tensorflow.optimize.Adafactor
class XLAShardingAdafactorOptimizer(tf.train.Optimizer):
  """Adafactor optimizer for XLA sharding."""

  def __init__(
      self,
      multiply_by_parameter_scale=True,
      learning_rate=None,
      decay_rate=None,
      beta1=0.0,
      clipping_threshold=1.0,
      factored=True,
      epsilon1=1e-30,
      epsilon2=1e-3,
      min_dim_size_to_factor=128,
      use_locking=False,
      cond_is_finite=False,  # cl/295761665 and reduce_rms change
      name='Adafactor',
  ):
    """Construct a new Adafactor optimizer.

    See class comment.

    Args:
      multiply_by_parameter_scale: a boolean
      learning_rate: an optional Scalar.
      decay_rate: an optional Scalar.
      beta1: a float value between 0 and 1
      clipping_threshold: an optional float >= 1
      factored: a boolean - whether to use factored second-moment estimator for
        2d variables
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
      min_dim_size_to_factor: only factor accumulator if two tensor dimensions
        are at least this size.
      use_locking: No clue what this does.
      cond_is_finite: Check if Adafactor sufficient stats are finite on update.
      name: optimizer name.

    Raises:
      ValueError: if absolute_update_scale and relative_update_scale_fn are both
        present or both absent.
    """
    super().__init__(use_locking, name)
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    assert learning_rate is not None
    self._learning_rate = learning_rate
    assert decay_rate is not None
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2
    self._min_dim_size_to_factor = min_dim_size_to_factor
    self._cond_is_finite = cond_is_finite

  def _factored_dims(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.
    If we factor the accumulator, then this function returns a list of two
    tf.Dimensions to reduce over.  We always pick the two largest dimensions.
    If there are not two dimensions of size >= min_dim_size_to_factor, then we
    do not factor.

    Args:
      shape: a Shape

    Returns:
      either a list of 2 Dimension indices or None
    """
    if not self._factored or len(shape) < 2:
      return None
    s = [(s, i) for i, s in enumerate(shape)]
    sorted_dims = sorted(s, key=lambda d: -d[0])
    if sorted_dims[1][0] < self._min_dim_size_to_factor:
      return None
    return sorted_dims[0][1], sorted_dims[1][1]

  def _parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.

    Returns:
      a Scalar
    """
    return tf.math.maximum(
        py_utils.ReduceRms(var), tf.constant(self._epsilon2, var.dtype))

  def _create_slots(self, var_list):
    for var in var_list:
      grad_dtype = var.dtype  # TODO(lepikhin): add to params
      if self._beta1:
        self._zeros_slot(var, 'm', self._name)

      factored_dims = self._factored_dims(var.shape.as_list())
      if factored_dims:
        d0, d1 = factored_dims
        vr_shape = copy.deepcopy(var.shape.as_list())
        del vr_shape[d0]
        vc_shape = copy.deepcopy(var.shape.as_list())
        del vc_shape[d1]
        r_val = tf.zeros(vr_shape, dtype=grad_dtype)
        c_val = tf.zeros(vc_shape, dtype=grad_dtype)
        self._get_or_make_slot(var, r_val, 'vr', self._name)
        self._get_or_make_slot(var, c_val, 'vc', self._name)
        tf.logging.info('Adafactor %s %r slot_vc: %r slot_vr: %r', var.name,
                        var.shape.as_list(), vc_shape, vr_shape)
      else:
        v_val = tf.zeros(var.shape, dtype=grad_dtype)
        self._get_or_make_slot(var, v_val, 'v', self._name)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_dense(
        tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
        handle)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  # _apply_dense simulation for testing purposes
  def try_apply_dense(self, grad, var):
    assert grad is not None

    cond = tf.constant(True)
    is_finite_checks = []
    stats = {}

    grad_dtype = var.dtype  # TODO(lepikhin): add to params
    grad = tf.cast(grad, grad_dtype)
    factored_dims = self._factored_dims(var.shape.as_list())
    if factored_dims:
      vr = self.get_slot(var, 'vr')
      vc = self.get_slot(var, 'vc')
    else:
      v = self.get_slot(var, 'v')
    if self._beta1:
      m = self.get_slot(var, 'm')

    def _Upd(c, k, x):
      stats[k] = x
      is_finite_checks.append(tf.reduce_all(tf.math.is_finite(x)))
      return c

    with tf.variable_scope(var.name[:-2] + '/Adafactor'):
      grad_squared = tf.math.square(grad) + tf.cast(self._epsilon1, grad_dtype)
      cond = _Upd(cond, 'grad_squared', grad_squared)  # 0 (factored)
      decay_rate = tf.cast(self._decay_rate, var.dtype)
      old_val = tf.identity(var)  # TODO(lepikhin): introduce gradient dtype
      assert self._multiply_by_parameter_scale
      if self._multiply_by_parameter_scale:
        parameter_scale = self._parameter_scale(old_val)
        cond = _Upd(cond, 'parameter_scale', parameter_scale)  # 1 (factored)
        update_scale = self._parameter_scale(old_val) * tf.cast(
            self._learning_rate, grad_dtype)

      else:
        update_scale = self._learning_rate
      mixing_rate = tf.cast(1.0 - decay_rate, grad_dtype)
      update_scale = tf.cast(update_scale, grad_dtype)
      if factored_dims:
        d0, d1 = factored_dims
        vr_axis, vc_axis = d0, d1
        grad_squared_row_mean = tf.reduce_mean(grad_squared, axis=vr_axis)
        grad_squared_col_mean = tf.reduce_mean(grad_squared, axis=vc_axis)
        # new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
        new_vr = vr * decay_rate + grad_squared_row_mean * mixing_rate
        # new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
        new_vc = vc * decay_rate + grad_squared_col_mean * mixing_rate
        cond = _Upd(cond, 'new_vr', new_vr)  # 2 (factored)
        cond = _Upd(cond, 'new_vc', new_vc)  # 3 (factored)
        # vr_update = _Wrap(tf.assign, vr, new_vr)
        # vc_update = _Wrap(tf.assign, vc, new_vc)
        # updates.extend([vr_update, vc_update])
        long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
        r_factor = tf.math.rsqrt(new_vr / long_term_mean)
        c_factor = tf.math.rsqrt(new_vc)
        mult = tf.expand_dims(r_factor, vr_axis) * tf.expand_dims(
            c_factor, vc_axis)
        cond = _Upd(cond, 'mult', mult)  # 4 (factored)
        x = grad * mult
      else:
        new_v = v * decay_rate + grad_squared * mixing_rate
        cond = _Upd(cond, 'new_v', new_v)
        # v_update = _Wrap(tf.assign, v, new_v)
        # updates.append(v_update)
        x = grad * tf.math.rsqrt(new_v)

      assert self._clipping_threshold is not None

      if self._clipping_threshold is not None:
        clipping_denom = tf.maximum(
            tf.constant(1.0, grad_dtype),
            py_utils.ReduceRms(x) /
            tf.constant(self._clipping_threshold, grad_dtype))
        x /= clipping_denom
      cond = _Upd(cond, 'x', x)
      subtrahend = x * update_scale
      if self._beta1:
        new_m = (
            m * tf.constant(self._beta1, dtype=grad_dtype) +
            subtrahend * tf.constant(1.0 - self._beta1, dtype=grad_dtype))
        subtrahend = new_m
        cond = _Upd(cond, 'new_m', new_m)
        # updates.append(_Wrap(tf.assign, m, new_m))

      # It is critical to use assign_sub instead of tf.assign(var - subtrahend)
      #  for the case of bfloat16 activations, so as to avoid repeatedly
      #  rounding the slice value, which results in poor quality.
      cond = _Upd(cond, 'subtrahend', subtrahend)  # 5 (factored)

      # var_update = _Wrap(tf.assign_sub, var, subtrahend)
      # updates.append(var_update)

      return is_finite_checks, stats

  def _resource_apply_dense(self, grad, var):
    if grad is None:
      tf.logging.warning('Gradient is None for variable %s' % var.name)
      return []

    grad_dtype = var.dtype  # TODO(lepikhin): add to params
    grad = tf.cast(grad, grad_dtype)
    factored_dims = self._factored_dims(var.shape.as_list())
    if factored_dims:
      vr = self.get_slot(var, 'vr')
      vc = self.get_slot(var, 'vc')
    else:
      v = self.get_slot(var, 'v')
    if self._beta1:
      m = self.get_slot(var, 'm')

    cond = tf.constant(True)

    def _Upd(c, x):
      if not self._cond_is_finite:
        return c
      c = tf.math.logical_and(c, tf.reduce_all(tf.math.is_finite(x)))
      c = tf.math.logical_and(
          c, tf.reduce_all(tf.math.logical_not(tf.math.is_inf(x))))
      return c

    def _Wrap(fn, x, y):
      if not self._cond_is_finite:
        return fn(x, y)
      return tf.cond(cond, lambda: fn(x, y), lambda: x)

    with tf.variable_scope(var.name[:-2] + '/Adafactor'):
      grad_squared = tf.math.square(grad) + tf.cast(self._epsilon1, grad_dtype)
      cond = _Upd(cond, grad_squared)
      decay_rate = tf.cast(self._decay_rate, var.dtype)
      old_val = tf.identity(var)  # TODO(lepikhin): introduce gradient dtype
      if self._multiply_by_parameter_scale:
        update_scale = self._parameter_scale(old_val) * tf.cast(
            self._learning_rate, grad_dtype)
      else:
        update_scale = self._learning_rate
      mixing_rate = tf.cast(1.0 - decay_rate, grad_dtype)
      update_scale = tf.cast(update_scale, grad_dtype)
      updates = []
      if factored_dims:
        d0, d1 = factored_dims
        vr_axis, vc_axis = d0, d1
        grad_squared_row_mean = tf.reduce_mean(grad_squared, axis=vr_axis)
        grad_squared_col_mean = tf.reduce_mean(grad_squared, axis=vc_axis)
        # new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
        new_vr = vr * decay_rate + grad_squared_row_mean * mixing_rate
        # new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
        new_vc = vc * decay_rate + grad_squared_col_mean * mixing_rate
        cond = _Upd(cond, new_vr)
        cond = _Upd(cond, new_vc)
        vr_update = _Wrap(tf.assign, vr, new_vr)
        vc_update = _Wrap(tf.assign, vc, new_vc)
        updates.extend([vr_update, vc_update])
        long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
        r_factor = tf.math.rsqrt(new_vr / long_term_mean)
        c_factor = tf.math.rsqrt(new_vc)
        x = grad * tf.expand_dims(r_factor, vr_axis) * tf.expand_dims(
            c_factor, vc_axis)
      else:
        new_v = v * decay_rate + grad_squared * mixing_rate
        cond = _Upd(cond, new_v)
        v_update = _Wrap(tf.assign, v, new_v)
        updates.append(v_update)
        x = grad * tf.math.rsqrt(new_v)
      if self._clipping_threshold is not None:
        clipping_denom = tf.maximum(
            tf.constant(1.0, grad_dtype),
            py_utils.ReduceRms(x) /
            tf.constant(self._clipping_threshold, grad_dtype))
        x /= clipping_denom
      subtrahend = x * update_scale
      if self._beta1:
        new_m = (
            m * tf.constant(self._beta1, dtype=grad_dtype) +
            subtrahend * tf.constant(1.0 - self._beta1, dtype=grad_dtype))
        subtrahend = new_m
        cond = _Upd(cond, new_m)
        updates.append(_Wrap(tf.assign, m, new_m))
      # It is critical to use assign_sub instead of tf.assign(var - subtrahend)
      #  for the case of bfloat16 activations, so as to avoid repeatedly
      #  rounding the slice value, which results in poor quality.
      cond = _Upd(cond, subtrahend)
      var_update = _Wrap(tf.assign_sub, var, subtrahend)
      updates.append(var_update)
      return tf.group(*updates)


class XLAShardingAdafactor(Base):
  """Adafactor optimizer for XLA sharding."""

  @classmethod
  def Params(cls):
    params = super().Params()
    params.Define('beta1', 0, 'Beta1 of Adam. Can be zero.')
    params.Define('beta2', 0.999, 'Beta2 of Adam.')
    params.Define(
        'multiply_by_parameter_scale', True,
        'If True, then compute absolute_update_scale as described '
        'in tensor2tensor/utils/adafactor.py. If False, let '
        'absolute_update_scale be the externally supplied '
        'learning_rate.')
    params.Define('clipping_threshold', None,
                  'Should be >=1.0 or None for no update clipping')
    params.Define(
        'factored', True,
        'Whether to factor the second-moment estimator. True means '
        'less memory usage.')
    params.Define('decay_exponent_pow', None,
                  'if set, call adafactor_decay_rate_pow from T2T adafactor')
    params.Define('decay_exponent_offset', 0,
                  'start step number for adafactor decay schedule')
    params.Define(
        'min_dim_size_to_factor', 128, 'only factor accumulator if '
        'two tensor dimensions are at least this size.')
    params.Define(
        'cond_is_finite', False,
        'Condition Adafactor update on sufficient stats being finite.')
    params.name = 'Adafactor'
    return params

  def GetOptimizer(self, lr):
    params = self.params
    if params.decay_exponent_pow:
      decay_rate = _AdafactorDecayRatePow(
          params.decay_exponent_pow, offset=params.decay_exponent_offset)
    else:
      decay_rate = _AdafactorDecayRateAdam(params.beta2)

    return XLAShardingAdafactorOptimizer(
        learning_rate=lr,
        factored=params.factored,
        clipping_threshold=params.clipping_threshold,
        multiply_by_parameter_scale=params.multiply_by_parameter_scale,
        decay_rate=decay_rate,
        beta1=params.beta1,
        min_dim_size_to_factor=params.min_dim_size_to_factor,
        cond_is_finite=params.cond_is_finite,
        name=params.name)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('adafactor_lr', lr)


class GradientAggregationOptimizer(tf.train.Optimizer):
  """Optimizer wrapper providing deferred gradient application.

  Large hardware configurations are in high-demand, and difficult to come by.
  This class enables simulating execution on large hardware configurations, by
  accumulating gradients across multiple batches. A few caveats apply:

  Batch statistics (e.g. batch norm) will continue to be based on the
  "micro-batch" size.
  This effectively trades off computation/time: simulating a large cluster on
  a single core will take an excessive amount of time.

  N.B. Learning rate schedules may need to be adjusted in addition to using
  this optimizer. Schedules should either be scaled down by the relative batch
  size, or use a schedule based on the number of examples to be consistent
  across different batch sizes.
  """

  def __init__(self, opt, num_micro_batches=1, apply_crs_to_grad=False):
    self._opt = opt
    self._num_micro_batches = num_micro_batches
    self._counter = None
    self._apply_crs_to_grad = apply_crs_to_grad

  def _create_slots(self, var_list):
    if not self._counter:
      self._counter = tf.get_variable(
          shape=[], initializer=tf.zeros_initializer, name='update_count')

    for v in var_list:
      vo = self._opt._zeros_slot(v, 'grad_accum', 'GradientAccumulator')  # pylint: disable=protected-access
      sharding = None
      try:
        sharding = gshard_utils.GetVarSharding(v)
      except ValueError:
        continue
      if sharding and not sharding.is_replicated:
        sharding.ApplyToVariable(vo)

  def compute_gradients(self, loss, var_list=None, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if self._num_micro_batches == 1:
      return self._opt.apply_gradients(grads_and_vars, global_step)
    global_step = global_step or py_utils.GetOrCreateGlobalStepVar()
    with tf.init_scope():
      self._create_slots([v for (_, v) in grads_and_vars])

    accums = []
    variables = []

    for g, v in grads_and_vars:
      accum = self.get_slot(v, 'grad_accum')
      variables.append(v)
      # pytype: disable=attribute-error
      if isinstance(g, tf.IndexedSlices):
        scaled_grad = tf.IndexedSlices(
            g.values / self._num_micro_batches,
            g.indices,
            dense_shape=g.dense_shape)
      else:
        scaled_grad = g / self._num_micro_batches
      accum_tensor = accum.read_value()
      accums.append(accum.assign(accum_tensor + scaled_grad))
      # pytype: enable=attribute-error

    def _ApplyAndReset():
      normalized_accums = accums
      if self._apply_crs_to_grad:
        normalized_accums = [
            tf.tpu.cross_replica_sum(accum.read_value()) for accum in accums
        ]
      apply_op = self._opt.apply_gradients(
          list(zip(normalized_accums, variables)))
      with tf.control_dependencies([apply_op]):
        zero_op = [tf.assign(accum, tf.zeros_like(accum)) for accum in accums]
      return tf.group(zero_op, tf.assign_add(global_step, 1))

    def _Accum():
      return tf.no_op()

    accum_step = tf.cond(
        tf.equal(
            tf.math.floormod(self._counter + 1, self._num_micro_batches), 0),
        _ApplyAndReset,  # Apply the accumulated gradients and reset.
        _Accum)  # Accumulate gradients.

    with tf.control_dependencies([tf.group(accums)]):
      return tf.group(accum_step, tf.assign_add(self._counter, 1))

  def get_slot(self, *args, **kwargs):
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    return self._opt.get_slot_names(*args, **kwargs)

  def variables(self):
    return self._opt.variables()


class XLAShardingAdafactorAccuGrad(XLAShardingAdafactor):
  """Adafactor optimizer that does gradient accumulation over N steps."""

  @classmethod
  def Params(cls):
    params = super().Params()
    params.Define('num_micro_batches', 1, 'Number of accumulated batches.')
    return params

  def GetOptimizer(self, lr):
    params = self.params
    optimizer = super().GetOptimizer(lr)
    if params.num_micro_batches > 1:
      tf.logging.info('Applying gradient aggregation.')
      optimizer = GradientAggregationOptimizer(
          optimizer, params.num_micro_batches, apply_crs_to_grad=True)
    return optimizer
