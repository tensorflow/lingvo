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
from lingvo.jax import optimizer_prefix_vectorization as opt_vec
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import pytypes
import optax
import tensorflow.compat.v2 as tf

JTensor = jnp.ndarray
NestedMap = py_utils.NestedMap
NestedJTensor = base_layer.NestedJTensor
NestedBool = base_layer.NestedBool
NestedParams = pytypes.NestedParams
InstantiableParams = py_utils.InstantiableParams


class Learner:
  """A learner."""

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Returns the Learner params."""
    p = InstantiableParams(cls)
    p.Define('name', '',
             'Name of this learner object, must be a valid identifier.')
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
    p.Define(
        'grad_norm_individual_vars', False,
        'Whether or not to export grad_norm for each individual variable'
        ' as summaries.')
    p.Define(
        'vectorize_on_repeat_prefix', True,
        'Whether to vectorize optimizers on the repeat_prefix dims of the '
        'variables. This allows stacking variables of different layers while '
        'not affecting the behavior of optimizers like Adafactor.')
    p.Define(
        'skip_step_gradient_norm_value', 0.0,
        'If non-zero, we skip a step entirely if gradient_norm exceeds '
        'this value.')
    p.Define('enable_skip_step_on_gradient_anomalies', True,
             'Skips the step if gradient anomaly (NaN/Inf) is detected.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructor for the learner."""
    assert params.name, ('Learner params for %s must have a "name"' %
                         self.__class__.__name__)
    module_name = params.name
    NestedMap.CheckKey(module_name)

    self._params = params.Copy()

    p = self.params
    asserts.not_none(p.optimizer)
    asserts.not_none(p.loss_name)
    self._optimizer = p.optimizer.Instantiate()
    self._grad_tx = self.optimizer.get_grad_transformation()

  @property
  def params(self) -> InstantiableParams:
    """Returns the params upon which this layer is built."""
    return self._params

  @property
  def optimizer(self) -> optimizers.BaseOptimizer:
    """Return the Optimizer object of this learner."""
    return self._optimizer

  def get_grad_tx(
      self, var_weight_params: NestedParams
  ) -> optimizers.GeneralGradientTransformation:
    # Apply vectorization on prefix dims.
    if not self.params.vectorize_on_repeat_prefix:
      return self._grad_tx
    return opt_vec.get_transformations_with_vectorized_repeat_prefix(
        self._grad_tx, var_weight_params)

  def scale_gradients(self, raw_grads: NestedMap) -> Tuple[NestedMap, JTensor]:
    """Scales the gradient.

    Args:
      raw_grads: A nested structure of gradient values.

    Returns:
     A nested structure with the rescaled gradient values.
     A predicate tensor indicating whether the step is valid, i.e., it does not
       have anomaly detected (e.g. Nan or Inf, or excessively big gradient norm)
       and should not be skipped.
    """
    p = self.params
    learner_name = self.params.name
    # Compute gradient norm.
    grad_squared = jax.tree_map(lambda x: jnp.sum(x * x), raw_grads)

    if p.grad_norm_individual_vars:
      grad_norms = jax.tree_map(jnp.sqrt, grad_squared)
      var_keys = py_utils.extract_prefixed_keys_from_nested_map(grad_norms)

      def add_grad_norm_summary(key, value):
        base_layer.add_summary(f'{learner_name}/var_grad_norm/{key}', value)

      jax.tree_map(add_grad_norm_summary, var_keys, grad_norms)

    grad_squared, _ = jax.tree_flatten(grad_squared)
    grad_squared = jnp.concatenate([x[jnp.newaxis] for x in grad_squared])
    raw_grad_norm = jnp.sqrt(jnp.sum(grad_squared))
    base_layer.add_summary(f'{learner_name}/grad_norm', raw_grad_norm)

    def keep_step(grad_norm):
      keep_threshold = p.skip_step_gradient_norm_value
      if keep_threshold:
        return jnp.logical_and(
            jnp.all(jnp.isfinite(grad_norm)),
            jnp.all(jnp.less(grad_norm, keep_threshold)))
      else:
        return jnp.all(jnp.isfinite(grad_norm))

    def clip_grads(grads, grad_norm):
      if p.optimizer.clip_gradient_norm_to_value:
        assert p.optimizer.clip_gradient_single_norm_to_value == 0.
        grad_scale = jnp.minimum(
            jnp.array(1, grad_norm.dtype),
            jnp.array(p.optimizer.clip_gradient_norm_to_value, grad_norm.dtype)
            / grad_norm)
        grads = jax.tree_map(lambda g: g * grad_scale, grads)
      elif p.optimizer.clip_gradient_single_norm_to_value:
        assert p.optimizer.clip_gradient_norm_to_value == 0.
        grad_single_norm = jax.tree_map(lambda x: jnp.sqrt(jnp.sum(x * x)),
                                        grads)

        def scale_gradient(grad, norm):
          return grad * jnp.minimum(
              jnp.array(1, norm.dtype),
              jnp.array(p.optimizer.clip_gradient_single_norm_to_value,
                        norm.dtype) / norm)

        grads = jax.tree_map(scale_gradient, grads, grad_single_norm)
        grad_scale = jnp.array(1.0)
      else:
        # no clipping is needed.
        grad_scale = jnp.array(1.0)
      return grads, grad_scale

    # Mark the step as invalid if any gradient anomaly is detected (e.g. Nan or
    # Inf, or excessively big gradient norm).
    valid_step = keep_step(raw_grad_norm)
    grads, grad_scale = clip_grads(raw_grads, raw_grad_norm)
    base_layer.add_summary('grad_scale', grad_scale)

    return grads, valid_step

  def update_states(
      self, grads: NestedMap, states: optax.OptState, old_vars: NestedJTensor,
      var_weight_params: NestedParams) -> Tuple[NestedMap, optax.OptState]:
    """Applies gradient transformation, updates optimizer states.

    Args:
      grads: A nested structure of gradient values.
      states: Optimizer states.
      old_vars: Current model weights.
      var_weight_params: Weight params of the vars.

    Returns:
      transformed_grad, new_states pair.
    """
    grads, valid_step = self.scale_gradients(grads)
    transformed_grad, new_states = self.get_grad_tx(var_weight_params).update(
        grads, states, old_vars)

    if self.params.enable_skip_step_on_gradient_anomalies:
      # Set grads to 0 if the step is invalid.
      transformed_grad = jax.tree_map(
          lambda x: jnp.where(valid_step, x, jnp.zeros_like(x)),
          transformed_grad)
      # Keep the old state if the step is invalid.
      new_states = jax.tree_map(lambda x, y: jnp.where(valid_step, x, y),
                                new_states, states)
    return transformed_grad, new_states

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

    # Add a summary of total var norm.
    var_squared = jax.tree_map(lambda x: jnp.sum(x * x), old_vars)
    var_squared, _ = jax.tree_flatten(var_squared)
    var_squared = jnp.concatenate([x[jnp.newaxis] for x in var_squared])
    var_norm = jnp.sqrt(jnp.sum(var_squared))
    base_layer.add_summary('var_norm', var_norm)

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


class MultiOptimizerLearner(Learner):
  """Multi-optimizer learner which supports multiple optimizers.

  This class assumes a primary optimizer as in the Learner class. To add more
  optimizers, call the `MultiOptimizerLearner` class with a list of
  `auxiliary_optimizers` which are applied on variables matched using the list
  `auxiliary_regex`, which will walk the PyTree of variables, and set all
  descendants to use the corresponding auxiliary optimizer. Note that the length
  of `auxiliary_optimizers` and `auxiliary_regex` must be the same.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:  # pylint: disable=invalid-name
    """Returns the Learner params."""
    p = super().Params()
    p.Define(
        'auxiliary_optimizers', [], 'Additional auxiliary optimizers for'
        'optimizing a subset of model variables.')
    p.Define(
        'auxiliary_regex', [], 'A regular expression which if matches'
        'the variable name, will activate the corresponding auxiliary'
        'optimizer. The length of this list must be the same as auxiliary'
        'optimiers.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    """Constructor for the MultiOptimizer learner."""
    super().__init__(params)
    p = self.params
    asserts.not_none(p.optimizer)
    asserts.not_none(p.loss_name)
    if len(p.auxiliary_optimizers) != len(p.auxiliary_regex):
      raise ValueError('The length of the auxiliary regex must match the length'
                       'of the auxiliary optimizers.')
    self._optimizer = p.optimizer.Instantiate()
    self._auxiliary_optimizers = [
        opt.Instantiate() for opt in p.auxiliary_optimizers
    ]
    self._grad_tx_fn = self._optimizer.get_grad_transformation
    self._auxiliary_grad_tx_fn = [
        opt.get_grad_transformation for opt in self._auxiliary_optimizers
    ]

  def get_grad_tx(
      self, var_weight_params: NestedParams
  ) -> optimizers.GeneralGradientTransformation:
    """The gradient transformation the MultiOptimizer lerner.

    Args:
      var_weight_params: The model vars' params which will be used to filter
        using the regex to determine which optimizer will be applied to which
        variable.

    Returns:
      Optax sharded gradient transformation.
    """
    p = self.params
    optimizer_chain = []
    optimizer_mask = []

    # Aggregate all the auxiliary optimizer masks.
    for regex, grad_tx_fn in zip(p.auxiliary_regex, self._auxiliary_grad_tx_fn):
      prefix = py_utils.extract_prefixed_keys_from_nested_map(var_weight_params)
      mask = jax.tree_map(lambda x, regex=regex: regex in x, prefix)
      optimizer_chain.append(optimizers.sharded_masked(grad_tx_fn(), mask))
      optimizer_mask.append(mask)

    # Create the default optimizer mask.
    def check_var_in_auxiliary_regex(*args):
      """Check if a variable is already activated by an auxiliary optimizer."""
      r = False
      for x in args:
        if x:
          # Check if it was already activated by some other optimizer.
          if r:
            raise ValueError('The regex pattern of auxiliary optimizers should'
                             'be non-overlapping.')
          r = True
      return r

    default_mask = jax.tree_map(check_var_in_auxiliary_regex,
                                     *optimizer_mask)
    default_mask = jax.tree_map(lambda mask: not mask, default_mask)
    optimizer_chain.insert(
        0, optimizers.sharded_masked(self._grad_tx_fn(), default_mask))
    grad_tx = optimizers.sharded_chain(*optimizer_chain)
    # Finally, apply vectorization on prefix dims.
    if p.vectorize_on_repeat_prefix:
      grad_tx = opt_vec.get_transformations_with_vectorized_repeat_prefix(
          grad_tx, var_weight_params)
    return grad_tx

  def update_states(
      self, grads: NestedMap, states: optax.OptState, old_vars: NestedJTensor,
      var_weight_params: NestedParams) -> Tuple[NestedMap, optax.OptState]:
    """Applies gradient transformation, updates optimizer states.

    Args:
      grads: A nested structure of gradient values.
      states: Optimizer states.
      old_vars: Current model weights.
      var_weight_params: Weight params of the vars.

    Returns:
      transformed_grad, new_states pair.
    """
    grads, valid_step = self.scale_gradients(grads)
    grad_tx = self.get_grad_tx(var_weight_params)
    transformed_grad, new_states = grad_tx.update(grads, states, old_vars)
    if self.params.enable_skip_step_on_gradient_anomalies:
      # Set grads to 0 if the step is invalid.
      transformed_grad = jax.tree_map(
          lambda x: jnp.where(valid_step, x, jnp.zeros_like(x)),
          transformed_grad)
      # Keep the old state if the step is invalid.
      new_states = jax.tree_map(lambda x, y: jnp.where(valid_step, x, y),
                                new_states, states)
    return transformed_grad, new_states
