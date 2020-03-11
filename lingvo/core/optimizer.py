# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import adagraft
from lingvo.core import base_layer
from lingvo.core import distributed_shampoo
from lingvo.core import py_utils
from lingvo.core import summary_utils


class Base(base_layer.BaseLayer):
  """Base class for all optimizers."""

  @classmethod
  def Params(cls):
    p = super(Base, cls).Params()
    p.name = cls.__name__
    return p

  def GetOptimizer(self, lr):
    """Returns the TF optimizer object."""
    raise NotImplementedError('Abstract method')

  def AddSummary(self, lr, optimizer, var_grad):
    """Adds summary if needed."""
    pass

  def ComputeGradients(self, loss, vmap, *args, **kwargs):
    """Allows subclasses control computation of gradients."""
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
    optimizer = self.GetOptimizer(lr)

    def _Apply():
      return optimizer.apply_gradients(
          [(g, v) for (v, g) in var_grad.Flatten()], name='meta_backprop')

    if not py_utils.use_resource_variables():
      var_update_op = _Apply()
    else:
      # Many optimizers, e.g., Adam, Adagrad, etc., create
      # variables. We need to ensure name scope and variable scope are
      # cleared. Otherwise, tpu.batch_parallel does not work.
      with tf.name_scope(None):
        with tf.variable_scope(
            tf.VariableScope(
                use_resource=True, reuse=self.VarReuseForSlotVars())):
          var_update_op = _Apply()
    self.AddSummary(lr, optimizer, var_grad)
    return var_update_op

  def ApplyPostTrainingLoop(self, global_step):
    """Applies any computation to run after each tpu trainining loop.

    Args:
      global_step: Global step variable.

    Returns:
      Ops to run after training loop ends.
    """
    return tf.no_op()


class SGD(Base):
  """SGD."""

  def GetOptimizer(self, lr):
    return tf.train.GradientDescentOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar('sgd_lr', lr)


class Momentum(Base):
  """Momentum optimizer."""

  @classmethod
  def Params(cls):
    p = super(Momentum, cls).Params()
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
    p = super(RMSProp, cls).Params()
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
    p = super(Adagrad, cls).Params()
    p.Define('initial_accumulator_value', 1.0,
             "Adagrad's initial_accumulator_value.")
    return p

  def GetOptimizer(self, lr):
    p = self.params
    return tf.train.AdagradOptimizer(
        learning_rate=lr, initial_accumulator_value=p.initial_accumulator_value)

  def AddSummary(self, lr, optimizer, var_grad):
    p = self.params
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
    p = super(AdaDelta, cls).Params()
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
    p = super(Adam, cls).Params()
    p.Define('beta1', 0.9, 'Beta1 for Adam.')
    p.Define('beta2', 0.999, 'Beta2 for Adam.')
    p.Define('epsilon', 1e-6, 'Epsilon for Adam.')
    p.name = 'Adam'
    return p

  @staticmethod
  def ParamsA():
    """Convenient method for a commonly used Adam config."""
    return Adam.Params().Set(beta1=0.9, beta2=0.997, epsilon=1e-9)

  @staticmethod
  def ParamsB():
    """Convenient method for another commonly used Adam config."""
    return Adam.Params().Set(beta1=0.9, beta2=0.98, epsilon=1e-9)

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


class Accumulator(Base):
  """Gradient accumulator wrapper."""

  @classmethod
  def Params(cls):
    p = super(Accumulator, cls).Params()
    p.Define('optimizer_tpl', Adam.Params(),
             'Params for the wrapped optimizer.')
    p.Define(
        'accum_steps', 5, 'Number of gradient accumulation steps'
        ' before invoking wrapped optimizer.')
    p.name = 'Accumulator'
    return p

  def __init__(self, params):
    super(Accumulator, self).__init__(params)
    p = self.params
    self.CreateChild('_opt', p.optimizer_tpl)

  def Apply(self, lr, var_grad):
    p = self.params

    def _Acc(vg):
      """Updating accumulators."""

      v, g = vg
      with tf.variable_scope(v.op.name):
        _, a = py_utils.CreateVariable(
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

    return tf.cond(
        tf.equal(
            tf.mod(self.theta.global_step, p.accum_steps), p.accum_steps - 1),
        _ApplyAndReset, lambda: tf.group(tf.no_op()))

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
    params = super(DistributedShampoo, cls).Params()
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
        global_step=self.theta.global_step)

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

    if not py_utils.use_resource_variables():
      var_update_op = _Apply()
    else:
      # Many optimizers, e.g., Adam, Adagrad, etc., create
      # variables. We need to ensure name scope and variable scope are
      # cleared. Otherwise, tpu.batch_parallel does not work.
      with tf.name_scope(None):
        with tf.variable_scope(
            tf.VariableScope(
                use_resource=True, reuse=self.VarReuseForSlotVars())):
          var_update_op = _Apply()
    self.AddSummary(lr, self._optimizer, var_grad)
    return var_update_op

  def ApplyPostTrainingLoop(self, global_step):
    """Applies any computation to run after each tpu trainining loop.

    Args:
      global_step: Global step variable.

    Returns:
      Ops to run after training loop ends.
    """
    invoke_async_ops = self._optimizer.invoke_async_preconditioner_computation(
        tf.cast(global_step, tf.int32))
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
    params = super(AdaGraft, cls).Params()

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
