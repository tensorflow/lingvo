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
"""Define the entmax sampling and entmax loss.

This is the entmax that demonstrated in this publication:
https://arxiv.org/pdf/2004.02644.pdf. And the implementation is based on the
https://github.com/deep-spin/entmax which is implemented in pytorch.
We hope to use it in Meena2 to unify the training and inference under the entmax
framework that can produce the sparse probabilities.
"""

from lingvo import compat as tf


def _calculate_probability(inputs: tf.Tensor, alpha: float = 1.5):
  """Calculate the probability."""
  # TODO(kfxiao): check the clip_value_max.
  return tf.math.pow(
      tf.clip_by_value(inputs, clip_value_min=0, clip_value_max=100),
      1.0 / (alpha - 1.0))


def entmax_support(inputs: tf.Tensor,
                   alpha: float = 1.5,
                   axis: int = -1,
                   n_iter: int = 50,
                   ensure_sum_one: bool = True) -> tf.Tensor:
  """Calculate the entmax probabilities."""

  @tf.custom_gradient
  def forward(inputs, alpha):
    with tf.name_scope("entmax_loss"):
      alpha_shape = inputs.get_shape().as_list()

      alpha_shape[axis] = 1
      alpha = tf.fill(alpha_shape, alpha)
      alpha = tf.cast(alpha, dtype=inputs.dtype)

      d = inputs.get_shape().as_list()[axis]
      alpha_m1 = alpha - 1.0

      inputs = inputs * alpha_m1

      max_val = tf.math.reduce_max(inputs, axis=axis, keepdims=True)
      tau_lo = max_val - tf.ones(
          alpha.get_shape().as_list(), dtype=inputs.dtype)
      tau_hi = max_val - tf.math.pow(
          tf.cast((1.0 / d), dtype=inputs.dtype), alpha_m1)

      f_lo = tf.math.reduce_sum(
          _calculate_probability(tf.math.subtract(inputs, tau_lo), alpha),
          axis) - 1.0

      dm = tau_hi - tau_lo

      for _ in range(n_iter):
        dm /= 2
        tau_m = tau_lo + dm
        p_m = _calculate_probability(inputs - tau_m, alpha)
        f_m = tf.math.reduce_sum(p_m, axis) - 1.0

        mask = tf.expand_dims(tf.math.greater(f_m * f_lo, 0), axis)
        tau_lo = tf.where(mask, tau_m, tau_lo)

      if ensure_sum_one:
        p_m /= tf.expand_dims(tf.math.reduce_sum(p_m, axis), axis)

    def grad_fn(d_outputs):
      with tf.name_scope("entmax_grad"):
        gppr = tf.where(p_m > 0, tf.math.pow(p_m, 2.0 - alpha),
                        tf.zeros_like(p_m))
        d_inputs = d_outputs * gppr
        q = tf.math.reduce_sum(d_inputs, axis) / tf.math.reduce_sum(gppr, axis)
        q = tf.expand_dims(q, axis)
        d_inputs -= q * gppr
        return d_inputs, d_inputs

    return p_m, grad_fn

  return forward(inputs, alpha)


def entmax_loss(labels: tf.Tensor,
                inputs: tf.Tensor,
                alpha: float = 1.5,
                n_iter: int = 50,
                ensure_sum_one: bool = True) -> tf.Tensor:
  """Calculates the loss using the entmax."""

  @tf.custom_gradient
  def forward(labels, inputs):
    with tf.name_scope("entmax_loss"):
      assert labels.get_shape().as_list()[0] == inputs.get_shape().as_list()[0]
      p_star = entmax_support(
          inputs, alpha=alpha, n_iter=n_iter, ensure_sum_one=ensure_sum_one)
      loss = (1.0 - tf.math.reduce_sum(tf.math.pow(p_star, alpha), axis=-1)) / (
          alpha * (alpha - 1))

      p_star -= tf.cast(labels, dtype=inputs.dtype)
      loss += tf.einsum("...IJ,...IJ->...I", p_star, inputs)

    def grad_fn(d_outputs):
      with tf.name_scope("entmax_loss_grad"):
        gradient = tf.expand_dims(d_outputs, -1) * p_star
        return gradient, gradient

    return loss, grad_fn

  return forward(labels, inputs)
