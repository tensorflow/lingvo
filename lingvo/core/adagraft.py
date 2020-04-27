# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""AdaGraft optimizer https://arxiv.org/abs/2002.11803 ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf


class AdaGraftOptimizer(tf.train.Optimizer):
  """Optimizer which combines per-layer direction and magnitude from two optimizers.

  Disentangling Adaptive Gradient Methods from Learning Rates
  Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, Cyril Zhang
  https://arxiv.org/abs/2002.11803
  """

  def __init__(self,
               learning_rate,
               magnitude_optimizer,
               direction_optimizer,
               diagnostic=False,
               use_global_norm=False,
               name="AdaGraft"):
    """Construct a new AdaGraft optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
      magnitude_optimizer: Child Optimizer to inherit step sizes.
      direction_optimizer: Child Optimizer to inherit step directions.
      diagnostic: Whether to record per-tensor step norms.
      use_global_norm: Graft global l2 norms rather than per-layer.
      name: Optional name prefix for the operations created when applying
        gradients.
    """
    super(AdaGraftOptimizer, self).__init__(False, name)
    self._learning_rate = learning_rate
    self.magnitude_optimizer = magnitude_optimizer
    self.direction_optimizer = direction_optimizer
    self.diagnostic = diagnostic
    self.use_global_norm = use_global_norm

  def _create_slots(self, var_list):
    self.magnitude_optimizer._create_slots(var_list)  # pylint: disable=protected-access
    self.direction_optimizer._create_slots(var_list)  # pylint: disable=protected-access

    for v in var_list:
      with tf.ops.colocate_with(v):
        self._zeros_slot(v, "scratch_copy", self._name)
        if self.diagnostic or self.use_global_norm:
          self._get_or_make_slot(v, tf.constant(0.0), "m_step_norm", self._name)
          self._get_or_make_slot(v, tf.constant(0.0), "d_step_norm", self._name)

  def _prepare(self):
    self.magnitude_optimizer._prepare()  # pylint: disable=protected-access
    self.direction_optimizer._prepare()  # pylint: disable=protected-access

    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = tf.convert_to_tensor(
        learning_rate, name="learning_rate")

    if self.use_global_norm:  # create list of all vars for global _finish
      self._variables = []

  def _apply_dense(self, grad, var):  # pylint: disable=g-doc-args
    return self._internal_apply_dense(
        grad,
        var,
        self.magnitude_optimizer._apply_dense,  # pylint: disable=protected-access
        self.direction_optimizer._apply_dense)  # pylint: disable=protected-access

  def _resource_apply_dense(self, grad, var):
    return self._internal_apply_dense(
        grad,
        var,
        self.magnitude_optimizer._resource_apply_dense,  # pylint: disable=protected-access
        self.direction_optimizer._resource_apply_dense)  # pylint: disable=protected-access

  def _internal_apply_dense(self, grad, var, magnitude_optimizer_apply_fn,
                            direction_optimizer_apply_fn):  # pylint: disable=g-doc-args
    """Main optimization logic of AdaGraft, which calls the child optimizers.

    Args:
      grad: Tensor containing gradients.
      var: Tensor containing parameter values.
      magnitude_optimizer_apply_fn: Apply magnitude optimizer.
      direction_optimizer_apply_fn: Apply direction optimizer.

    Returns:
      The final update op, which increments var by the grafted step.

    Pseudocode:
    - Copy weights into scratch space 'scratch_copy'.
    - Run magnitude_optimizer in-place.
    - Use scratch copy to figure out how far we moved ('magnitude_step').
    - Copy weights back.
    - Run direction_optimizer in-place.
    - Move weights along the line segment with scratch_copy.
    """

    if self.use_global_norm:
      self._variables.append(var)

    # Slot with current parameter values
    scratch_slot = self.get_slot(var, "scratch_copy")
    old_var = tf.assign(scratch_slot, var)

    with tf.control_dependencies([old_var]):
      m_updated_var = magnitude_optimizer_apply_fn(grad, var)  # pylint: disable=protected-access

    # Run magnitude optimizer and compute the norm of the update.
    with tf.control_dependencies([m_updated_var]):
      m_step = var - old_var
      m_step_norm = tf.norm(m_step)
      if self.diagnostic or self.use_global_norm:
        m_step_norm = tf.assign(self.get_slot(var, "m_step_norm"), m_step_norm)

    # Run direction optimizer and compute its norm, and the direction.
    with tf.control_dependencies([m_step_norm]):
      flushed_var = tf.assign(var, old_var)
    with tf.control_dependencies([flushed_var]):
      d_updated_var = direction_optimizer_apply_fn(grad, var)  # pylint: disable=protected-access

    # Run an update of the direction optimizer with magnitude optimizer norm.
    with tf.control_dependencies([d_updated_var]):
      d_step = var - old_var
      d_step_norm = tf.norm(d_step)
      if self.diagnostic or self.use_global_norm:
        d_step_norm = tf.assign(self.get_slot(var, "d_step_norm"), d_step_norm)
      if self.use_global_norm:
        flushed_var = tf.assign(var, old_var)
        with tf.control_dependencies([d_step_norm, flushed_var]):
          return tf.assign(scratch_slot, d_step)
      step = tf.where(
          tf.greater(d_step_norm, 0),
          (m_step_norm / tf.maximum(d_step_norm, 1e-30)) * d_step,
          tf.zeros_like(d_step))
      return tf.assign(var, old_var + self._learning_rate_tensor * step)

  def _finish(self, update_ops, name_scope):
    with tf.control_dependencies(update_ops):
      ops1 = self.magnitude_optimizer._finish([], name_scope + "_m")  # pylint: disable=protected-access
      ops2 = self.direction_optimizer._finish([], name_scope + "_d")  # pylint: disable=protected-access

      if self.use_global_norm:  # apply global grafting
        with tf.control_dependencies([ops1, ops2]):
          m_global_norm = tf.Variable(0.)
          d_global_norm = tf.Variable(0.)
          for var in self._variables:
            m_step_norm = self.get_slot(var, "m_step_norm")
            d_step_norm = self.get_slot(var, "d_step_norm")
            tf.assign_add(m_global_norm, m_step_norm**2)
            tf.assign_add(d_global_norm, d_step_norm**2)

          multiplier = tf.sqrt(m_global_norm / tf.maximum(d_global_norm, 1e-30))

          step_ops = []
          for var in self._variables:
            d_step = self.get_slot(var, "scratch_copy")
            step = tf.where(
                tf.greater(d_step_norm, 0), multiplier * d_step,
                tf.zeros_like(d_step))
            step_op = tf.assign_add(var, self._learning_rate_tensor * step)
            step_ops.append(step_op)
          return tf.group(*step_ops, name=name_scope)

    return tf.group(*([ops1, ops2] + update_ops), name=name_scope)

  # Sparse gradients are not handled currently and is part of future work.
  def _resource_apply_sparse(self, grad_values, var, grad_indices):
    return tf.no_op()

  def _apply_sparse(self, grad, var):
    return tf.no_op()
