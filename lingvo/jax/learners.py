# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Module with the Learner class."""

from typing import Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import asserts
from lingvo.jax import base_layer
from lingvo.jax import optimizers
from lingvo.jax import py_utils
import optax
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
NestedBool = base_layer.NestedBool
InstantiableParams = py_utils.InstantiableParams


class Learner(base_layer.BaseLayer):
  """A learner."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Returns the Learner params."""
    p = super().Params()
    p.Define('loss_name', None,
             'Name of the loss this learner optimizes. Must not be None.')
    p.Define('optimizer', None, 'Params for the optimizer.')
    p.Define(
        'skip_zero_gradients', None,
        'If set, skips aggregating zero gradients while computing gradients.'
        'This helps in case where some weights may not be used in forward '
        'computation, e.g., sparsely activated networks or switchable layers '
        'in neural architectural search. '
        'Possible values are: '
        'None: do not skip zero gradients; '
        '"variable": skip if the entire variable gradients are almost zero.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructor."""
    super().__init__(params)
    p = self.params
    asserts.not_none(p.optimizer)
    asserts.not_none(p.loss_name)
    self._optimizer = p.optimizer.Instantiate()
    self._grad_tx = self._optimizer.get_grad_transformation()

  @property
  def optimizer(self) -> optimizers.BaseOptimizer:
    """Return the Optimizer object of this learner."""
    return self._optimizer

  @property
  def grad_tx(self) -> optax.GradientTransformation:
    return self._grad_tx

  def scale_gradients(self, grads: NestedMap) -> NestedMap:
    """Scales the gradient.

    Args:
      grads: A nested structure of gradient values.

    Returns:
     A nested structure with the rescaled gradient values.
    """
    p = self.params
    # Compute gradient norm.
    grad_squared = jax.tree_map(lambda x: jnp.sum(x * x), grads)
    grad_squared, _ = jax.tree_flatten(grad_squared)
    grad_squared = jnp.concatenate([x[jnp.newaxis] for x in grad_squared])
    grad_norm = jnp.sqrt(jnp.sum(grad_squared))
    learner_name = self.params.name
    base_layer.add_summary(f'{learner_name}/grad_norm', grad_norm)
    if p.optimizer.clip_gradient_norm_to_value:
      assert p.optimizer.clip_gradient_single_norm_to_value == 0.
      grad_scale = jnp.minimum(
          jnp.array(1, grad_norm.dtype),
          jnp.array(p.optimizer.clip_gradient_norm_to_value, grad_norm.dtype) /
          grad_norm)
      grads = jax.tree_map(lambda g: g * grad_scale, grads)
    elif p.optimizer.clip_gradient_single_norm_to_value:
      assert p.optimizer.clip_gradient_norm_to_value == 0.
      grad_single_norm = jax.tree_map(lambda x: jnp.sqrt(jnp.sum(x * x)), grads)

      def scale_gradient(grad, norm):
        return grad * jnp.minimum(
            jnp.array(1, grad_norm.dtype),
            jnp.array(p.optimizer.clip_gradient_single_norm_to_value,
                      grad_norm.dtype) / norm)

      grads = jax.tree_map(scale_gradient, grads, grad_single_norm)
    return grads

  def update_states(
      self, grads: NestedMap, states: optax.OptState,
      old_vars: NestedJTensor) -> Tuple[NestedMap, optax.OptState]:
    """Applies gradient transformation, updates optimizer states.

    Args:
      grads: A nested structure of gradient values.
      states: Optimizer states.
      old_vars: Current model weights.

    Returns:
      transformed_grad, new_states pair.
    """
    grads = self.scale_gradients(grads)
    return self._grad_tx.update(grads, states, old_vars)

  def apply_gradient(
      self,
      old_vars: NestedJTensor,
      transformed_grads: NestedJTensor,
      var_is_learnable: NestedBool,
  ) -> NestedJTensor:
    """Applies grads to model_variables.

    Note, in a flax model learnable variables are often referred to as 'params'.
    But since 'params' in Lingvo often refers to a hyperparams.Params, we
    refer to learnable weights of a network as 'variables'.

    Args:
      old_vars: a nested structure of model variables.
      transformed_grads: grads of loss wrt to the old_vars. Must be of the same
        structure as old_var. 'transformed_grads' have already gone through
        various gradient transformations.
      var_is_learnable: a nested structure of boolean values indicate whether a
        var is trainable. Must be of the same structure as old_vars.
        'non-trainable' vars include batch norm stats, various other counts,
        etc. Only learnable variables are updated.

    Returns:
      updated variables. Only learnable variables are updated.
    """
    p = self.params
    tf.nest.assert_same_structure(old_vars, transformed_grads)
    tf.nest.assert_same_structure(old_vars, var_is_learnable)

    assert p.skip_zero_gradients is None

    # TODO(yonghui): implement skip_zero_gradients.
    # TODO(yonghui): implement numerical checks.

    def _adjust_var(old_var, transformed_grad, is_learnable):
      if is_learnable:
        return old_var + transformed_grad
      else:
        return old_var

    return tf.nest.map_structure(_adjust_var, old_vars, transformed_grads,
                                 var_is_learnable)
    # TODO(yonghui): export gradient / variable summaries.

  @property
  def loss_name(self) -> str:
    return self._params.loss_name
