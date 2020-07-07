# Lint as: python3
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
"""Exponentiated Gradient Delta-Delta optimizer."""

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
# pylint: enable=g-direct-tensorflow-import


class EGDD(optimizer.Optimizer):
  """A version of GD Momentum with adaptive gain and learning rate.

  Exponentiated Gradient Delta-delta optimizer starts with a local gain of 1.0
  for every weight and a lr_scale of 1.0 for all weights. The EGDD update rule
  applies:

  momentum <- mu * momentum + learning_rate * gain * grad
  var  <- var - lr_scale * momentum

  The gain as well as the lr_scale are updated using the unnormalized
  exponentiated gradient algorithm [KW97].

  Reference: TBA

  [KW97] Kivinen, J., & Warmuth, M. K. Exponentiated gradient versus gradient
  descent for linear predictors. Information and Computation, 1997.
  """

  def __init__(self,
               learning_rate,
               momentum,
               beta=0.9,
               gain_learning_rate=0.01,
               scale_learning_rate=0.001,
               initial_gain=1.0,
               min_gain=1e-2,
               max_gain=1e2,
               initial_scale=1.0,
               min_scale=1e-1,
               max_scale=1e1,
               use_directions=True,
               use_signs=True,
               name="EGDD"):
    """Construct a new EG-DD optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value. The learning rate.
      momentum: A `Tensor` or a floating point value.
      beta: `float` decay rate of the gradient EMA.
      gain_learning_rate: `float` gain learning rate.
      scale_learning_rate: `float` scale learning rate.
      initial_gain: `float` initial gain.
      min_gain: `float` minimum gain.
      max_gain: `float` maximum gain,
      initial_scale: `float` initial scale.
      min_scale: `float` minimum learning rate scale.
      max_scale: `float` maximum learning rate scale.
      use_directions: `bool` whether to use directions only for scale updates.
      use_signs: `bool` whether to use the signs for updating gains.
      name: Optional name prefix for the operations created when applying
        gradients.

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.

    """
    super().__init__(False, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._beta = beta
    self._gain_learning_rate = gain_learning_rate
    self._scale_learning_rate = scale_learning_rate
    self._initial_gain = initial_gain
    self._min_gain = min_gain
    self._max_gain = max_gain
    self._initial_scale = initial_scale
    self._min_scale = min_scale
    self._max_scale = max_scale
    self._use_directions = use_directions
    self._use_signs = use_signs

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "momentum", self._name)
      self._zeros_slot(v, "gbar", self._name)
      g_tensor = ops.convert_to_tensor(v)
      gain_init = self._initial_gain * array_ops.ones_like(g_tensor)
      _ = self._get_or_make_slot(v, self._initial_scale * array_ops.ones((1)),
                                 "lr_scale", self._name)
      _ = self._get_or_make_slot(v, gain_init, "gain", self._name)
      _ = self._get_or_make_slot(v, array_ops.zeros((1)), "counter", self._name)

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")
    momentum = self._call_if_callable(self._momentum)
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")

  def _apply_dense(self, grad, var):
    lr_scale = self.get_slot(var, "lr_scale")
    momentum = self.get_slot(var, "momentum")
    gbar = self.get_slot(var, "gbar")
    gain = self.get_slot(var, "gain")
    counter = self.get_slot(var, "counter")
    counter_updated = state_ops.assign(counter, counter + 1)

    # lr_scale update uses normalized grad and momentum to be independent of dim
    normalized_grad = grad / (linalg_ops.norm(grad) + 1e-10)
    normalized_momentum = momentum / (linalg_ops.norm(momentum) + 1e-10)
    # Apply EG updates on lr_scale:
    # grad_lr_scale = -inner_product(current_grad, old_momentum)
    # lr_scale <- lr_scale * exp(-scale_learning_rate * grad_lr_scale)
    lr_scale_unnormalized_updated = clip_ops.clip_by_value(
        lr_scale * math_ops.exp(
            self._scale_learning_rate * math_ops.reduce_sum(grad * momentum)),
        self._min_scale, self._max_scale)
    lr_scale_normalized_updated = clip_ops.clip_by_value(
        lr_scale * math_ops.exp(self._scale_learning_rate * math_ops.reduce_sum(
            normalized_grad * normalized_momentum)), self._min_scale,
        self._max_scale)
    lr_scale_updated = state_ops.assign(
        lr_scale,
        array_ops.where(self._use_directions, lr_scale_normalized_updated,
                        lr_scale_unnormalized_updated))
    # remove the bias of zero initialization in gbar
    corrected_gbar = gbar / (
        1.0 - self._beta**math_ops.maximum(counter_updated - 1, 1))
    # Apply EG updates on gain:
    # grad_gain = - current_grad * old_gbar
    # gain <- gain * exp(-gain_learning_rate * grad_gain)
    gain_unnormalized_updated = clip_ops.clip_by_value(
        gain * math_ops.exp(self._gain_learning_rate * grad * corrected_gbar),
        self._min_gain, self._max_gain)
    # Normalized update uses sign(grad) * sign(gbar) as a proxy for grad_gain.
    gain_normalized_updated = clip_ops.clip_by_value(
        gain * math_ops.exp(self._gain_learning_rate * math_ops.sign(grad) *
                            math_ops.sign(gbar)), self._min_gain,
        self._max_gain)
    gain_updated = state_ops.assign(
        gain,
        array_ops.where(self._use_signs, gain_normalized_updated,
                        gain_unnormalized_updated))
    scaled_g = self._learning_rate_tensor * gain_updated * grad
    with ops.control_dependencies([lr_scale_updated, scaled_g]):
      momentum_updated = state_ops.assign(
          momentum, self._momentum_tensor * momentum + scaled_g)
      gbar_updated = state_ops.assign(
          gbar, self._beta * gbar + (1.0 - self._beta) * grad)
    with ops.control_dependencies([gbar_updated]):
      return state_ops.assign_sub(var, lr_scale_updated * momentum_updated)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  # Sparse gradients are not handled currently and is part of future work.
  def _resource_apply_sparse(self, grad_values, var, grad_indices):
    return control_flow_ops.no_op()

  def _apply_sparse(self, grad, var):
    return control_flow_ops.no_op()
