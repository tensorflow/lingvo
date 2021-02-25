# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""The Gradient Sign Dropout (GradDrop) algorithm."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import summary_utils


class GradDrop(base_layer.BaseLayer):
  """Implements the Gradient Sign Dropout (GradDrop) algorithm.

  This is most useful when your model computes a shared representation tensor
  that is then used by multiple separate downstream tasks. It is not applicable
  when the shared layer computes separate tensors for each downstream task. For
  example, if the inputs are different for each task, or if the sub-network is
  different for each task.

  To use this layer in your model, do the following steps:

  1. In your model, select a layer to insert GradDrop after; this is usually
     a layer that emits a shared representation that is then used by other
     task-specific layers. At that layer, apply the GradDrop layer to get
     an identity transformation of that feature. The GradDrop layer will
     modify the gradients that get backpropagated to the earlier layers.

  2. In your task at ComputeLoss, call the SetLosses function on the
     layer which you applied GradDrop to. You should also include the leak ratio
     parameters. To get access to that layer, you can manage the layer instance
     directly in your task, or simply recurse through all the children layer to
     find the GradDrop layer instance.

  [1] Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign
      Dropout. NeurIPS 2020. https://arxiv.org/abs/2010.06808
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'keep_prob_function', 'linear',
        'Linear or sigmoid transformation function for computing '
        'keep probability.')
    p.Define('keep_prob_function_scale', 1.0,
             'Scaling factor for keep_prob_function.')
    p.Define(
        'use_input_sign_only', True,
        'If True, this will compute the mask using only the sign (i.e., '
        'input / |input|.')
    p.Define(
        'keep_gradnorm_constant', True,
        'Whether to rescle the output of GradDrop so that the gradient '
        'norm is maintained.')
    p.Define('marginalize_batch_dim', True,
             'Whether to sum the gradient signal over the batch dimension.')
    p.Define('epsilon', 1e-7, 'Epsilon for numerical stability.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._losses = None
    self._output_tensor = None
    if p.keep_prob_function not in ['sigmoid', 'linear']:
      raise ValueError('keep_prob_function must be `sigmoid` or `linear`. '
                       'keep_prob_function=%s' % p.keep_prob_function)

  def SetLosses(self, losses):
    """Sets the losses.

    The leak ratio controls how much of the original gradient to pass through.

    In practice, we usually set leak_ratio to 0 for all the losses. However,
    in the transfer learning scenario where some task(s) loss is clearly more
    important than the other tasks -- one may choose to set the leak_ratio
    of the important task(s) to 1.0. This will pass through the gradient for
    those task(s) unchanged, and apply GradDrop only to the other losses.

    Args:
      losses: A list of tuples (loss, leak_ratio).
    """
    if self._losses is not None:
      raise ValueError('Losses already set.')
    tf.logging.info('Setting graddrop losses.')
    self._losses = losses

  def FProp(self, theta, input_tensor):
    p = self.params

    if self._output_tensor is not None:
      raise ValueError('FProp was already called.')

    def _Gradient(inputs, _, original_grad):

      # Compute the gradients for each loss w.r.t. the inputs.
      # TODO(jngiam): Look into whether TF dedups this computation.
      per_loss_grads = []
      for loss, _ in self._losses:
        per_loss_grad = tf.gradients(loss, self._output_tensor)[0]
        if per_loss_grad is None:
          tf.logging.warning(
              'Loss %s did not result in a gradient during '
              'GradDrop computation.', loss)
        else:
          per_loss_grads.append(per_loss_grad)

      if not per_loss_grads:
        raise ValueError('No valid gradients for GradDrop.')

      # Multiply the gradients with the inputs.
      grads = per_loss_grads
      if p.use_input_sign_only:
        input_abs = tf.abs(
            tf.cast(tf.abs(inputs) <= p.epsilon, tf.float32) + inputs)
        grads = [grad * ((inputs) / (input_abs)) for grad in grads]
      else:
        grads = [grad * inputs for grad in grads]

      # Sum gradient over batch, assuming that batch is always on dim 0.
      if p.marginalize_batch_dim:
        grads = [tf.reduce_sum(grad, axis=0, keepdims=True) for grad in grads]

      # First discretize all gradients into their sign values.
      grad_sign_positive = [tf.cast(grad > 0.0, tf.float32) for grad in grads]
      grad_sign_negative = [tf.cast(grad < 0.0, tf.float32) for grad in grads]

      # Calculate the probability of positive gradients based on equation (1)
      # in the GradDrop paper.
      grad_abs_sum = tf.add_n([tf.abs(grad) for grad in grads])
      prob_pos = (tf.add_n(grads) / (2. * grad_abs_sum + p.epsilon))
      # Implementation of different scales for the keep function. Larger
      # scales result in steeper keep functions.
      prob_pos *= p.keep_prob_function_scale

      if p.keep_prob_function == 'sigmoid':
        # Standard sigmoid has derivative of 0.25 at 0 so the factor of 4.0
        # allows the function scale in sigmoid to be compatible with the
        # function scale in the linear case.
        prob_pos = tf.sigmoid(4.0 * prob_pos)
      elif p.keep_prob_function == 'linear':
        prob_pos += 0.5

      # The main, default mode of GradDrop. Only gradients of one sign are kept,
      # and which sign is calculated via equation (1) of the main paper.
      prob_pos = tf.cast(prob_pos >= tf.random.uniform(prob_pos.shape),
                         tf.float32) - 0.5
      grad_masks = [(gsp - gsn) * prob_pos >= 0
                    for (gsn,
                         gsp) in zip(grad_sign_negative, grad_sign_positive)]

      # This diag value gives us the percentage of grads which are kept.
      gradmask_diag = [tf.cast(gm, tf.float32) for gm in grad_masks]
      diag = tf.reduce_mean(tf.add_n(gradmask_diag) / len(grad_masks))
      summary_utils.scalar('average_grad_mask', diag)
      leak_ratios = [leak_ratio for _, leak_ratio in self._losses]
      transformed_per_loss_grads = [
          grad * (leak + (1.0 - leak) * tf.cast(grad_mask, tf.float32))
          for (leak, grad,
               grad_mask) in zip(leak_ratios, per_loss_grads, grad_masks)
      ]

      transformed_grad = tf.cast(
          tf.add_n(transformed_per_loss_grads), original_grad.dtype)

      if not p.keep_gradnorm_constant:
        return transformed_grad

      transformed_grad_norm = tf.sqrt(tf.reduce_sum(transformed_grad**2))
      original_grad_norm = tf.sqrt(tf.reduce_sum(original_grad**2))
      return transformed_grad * original_grad_norm / (
          transformed_grad_norm + p.epsilon)

    output_tensor = py_utils.CallDefun(tf.identity, input_tensor, _Gradient)
    self._output_tensor = tf.identity(output_tensor)
    return self._output_tensor
