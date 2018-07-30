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

import tensorflow as tf

from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import summary_utils


class Base(object):
  """Base class for all optimizers."""

  @classmethod
  def Params(cls):
    """Returns the optimizer params."""
    p = hyperparams.Params()
    p.Define('cls', cls, 'Cls that this param object is associated with.')
    p.Define('add_summary', True, 'Adds summary iff true.')
    return p

  @property
  def params(self):
    return self._params

  def __init__(self, params):
    self._params = params.Copy()

  def GetOptimizer(self, lr):
    """Returns the TF optimizer object."""
    raise NotImplementedError('Abstract method')

  def AddSummary(self, lr, optimizer, var_grad):
    """Adds summary if needed."""
    pass

  def Apply(self, lr, var_grad):
    """Applies the gradient to the variable.

    Args:
      lr: A scalar. The base learning rate.
      var_grad: A NestedMap of (var, grad) pairs.

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
            tf.VariableScope(use_resource=True, reuse=False)):
          var_update_op = _Apply()
    self.AddSummary(lr, optimizer, var_grad)
    return var_update_op


class SGD(Base):
  """SGD."""

  def GetOptimizer(self, lr):
    return tf.train.GradientDescentOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar(self.params, 'sgd_lr', lr)


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
    summary_utils.scalar(self.params, 'momentum_lr', lr)


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
    summary_utils.scalar(p, 'adagrad_lr', lr)
    for v, _ in var_grad.Flatten():
      slot = optimizer.get_slot(v, 'accumulator')
      assert slot is not None
      summary_utils.scalar(p, 'optimizer/adagrad_accum_%s' % v.name,
                           tf.reduce_mean(slot))


class Adam(Base):
  """Adam."""

  @classmethod
  def Params(cls):
    p = super(Adam, cls).Params()
    p.Define('beta1', 0.9, 'Beta1 for Adam.')
    p.Define('beta2', 0.999, 'Beta2 for Adam.')
    p.Define('epsilon', 1e-6, 'Epsilon for Adam.')
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
        learning_rate=lr, beta1=p.beta1, beta2=p.beta2, epsilon=p.epsilon)

  def AddSummary(self, lr, optimizer, var_grad):
    summary_utils.scalar(self.params, 'adam_lr', lr)


class Accumulator(Base):
  """Gradient accumulator wrapper."""

  @classmethod
  def Params(cls):
    params = super(Accumulator, cls).Params()
    params.Define('dtype', tf.float32, 'Datatype to use.')
    params.Define('optimizer_tpl', Adam.Params(),
                  'Params for the wrapped optimizer.')
    params.Define(
        'accum_steps', 5, 'Number of gradient accumulation steps'
        ' before invoking wrapped optimizer.')
    return params

  def __init__(self, params):
    super(Accumulator, self).__init__(params)
    p = self.params
    self._opt = p.optimizer_tpl.cls(
        p.optimizer_tpl.Copy().Set(add_summary=p.add_summary))

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

      return v, a

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
            tf.mod(tf.train.get_or_create_global_step(), p.accum_steps),
            p.accum_steps - 1), _ApplyAndReset, lambda: tf.group(tf.no_op()))

  def GetOptimizer(self, lr):
    return self._opt.GetOptimizer(lr)

  def AddSummary(self, lr, optimizer, var_grad):
    return self._opt.AddSummary(lr, optimizer, var_grad)
