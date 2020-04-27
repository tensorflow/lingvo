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
"""An learner optimizes a subset of variables according to a loss.

It consists of a learning rate schedule, an optimizer, and gradient clipping
mechanisms. A BaseTask can have multiple learners, each optimizing a (usually
disjoint) subset of variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import summary_utils


class Learner(base_layer.BaseLayer):
  """A training program layer.

  The layer takes a loss tensor as input and outputs a trainer op.
  """

  @classmethod
  def Params(cls):
    p = super(Learner, cls).Params()
    p.Define(
        'l2_regularizer_weight', None,
        'If not None, L2 regularization to apply to the weights. '
        'Otherwise, disable L2 regularization.')
    p.Define(
        'l1_regularizer_weight', None,
        'If not None, L1 regularization to apply to the weights. '
        'Otherwise, disable L1 regularization.')
    p.Define('learning_rate', 0.0, 'learning rate to use.')
    p.Define(
        'clip_gradient_norm_to_value', 0.0,
        'Clip gradient by global norm to this value. This is similar to '
        'the bahaviour of tf.clip_by_global_norm, if you are looking for '
        'tf.clip_by_norm refer to clip_gradient_single_norm_to_value. Note '
        'these are mutually exclusive.')
    p.Define(
        'clip_gradient_single_norm_to_value', 0.0,
        'Clip gradient by single tensor norm to this value. This is '
        'similar to the bahaviour of tf.clip_by_norm. Note this is mutually '
        'exlusive to using clip_gradient_norm_to_value.')
    p.Define('grad_norm_to_clip_to_zero', 0.0,
             'Clip gradient to 0 if its norm exceeds this value.')
    p.Define('grad_norm_tracker', None, 'Params for GradNormTracker.')
    p.Define('optimizer', optimizer.Adam.Params(), 'Params for the optimizer.')
    p.Define('lr_schedule', schedule.ContinuousSchedule.Params(),
             'Learning rate decay schedule.')
    p.Define(
        'bprop_variable_filter', None,
        'If set, only backprop variables whose names partially match '
        'this regexp (re.search).')
    p.Define(
        'bprop_variable_exclusion', None,
        'If set, do not backprop variables whose names partially match '
        'this regexp (re.search).')
    p.Define(
        'grad_aggregation_method', tf.AggregationMethod.EXPERIMENTAL_TREE,
        'Specifies the method used to combine gradient terms. Accepted '
        'values are constants defined in the class AggregationMethod.')
    p.Define(
        'gate_gradients', False,
        'If True, add a tuple around the gradients returned for an '
        'operations. This avoids some race conditions.')
    p.Define('colocate_gradients_with_ops', True,
             'If True, try colocating gradients with the corresponding op.')
    p.Define(
        'skip_zero_gradients', None,
        'If set, skips aggregating zero gradients while computing gradients.'
        'This helps in case where some weights may not be used in forward '
        'computation, e.g., sparsely activated networks or switchable layers '
        'in neural architectural search. '
        'Possible values are: '
        'None: do not skip zero gradients; '
        '"variable": skip if the entire variable gradients are almost zero; '
        '"weight": skip if the individual weight gradients are almost zero.')
    p.Define('scale_gradients', True,
             'Whether to apply gradients adjustment and scaling.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(Learner, self).__init__(params)
    p = self.params

    self._var_grads = None
    self._eval_metrics = {}
    if p.grad_norm_tracker:
      # Use parent's name for backwards compatibility.
      with tf.variable_scope(self.parent.params.name):
        self.CreateChild('grad_norm_tracker', p.grad_norm_tracker)
    self.CreateChild('lr_schedule', p.lr_schedule)
    self.CreateChild('optimizer', p.optimizer)

  def GetVarGrads(self):
    return self._var_grads

  def GetTrainableVariables(self, vmap):
    p = self.params
    pos = re.compile(
        p.bprop_variable_filter) if p.bprop_variable_filter else None
    neg = re.compile(
        p.bprop_variable_exclusion) if p.bprop_variable_exclusion else None

    def VariableFilter(v):
      """Returns True if variable v should be optimized by this learner."""
      if pos and not pos.search(v.name):
        tf.logging.info('%s: disabled by bprop_variable_filter: %s',
                             p.name, v.name)
        return False
      if neg and neg.search(v.name):
        tf.logging.info('%s: disabled by bprop_variable_exclusion: %s',
                             p.name, v.name)
        return False
      return True

    return vmap.Filter(VariableFilter)

  def ApplyPostTrainingLoop(self, global_step):
    """Applies any computation to run after each tpu trainining loop.

    Args:
      global_step: Global step variable.

    Returns:
      Ops to run after training loop ends.
    """
    return self.optimizer.ApplyPostTrainingLoop(global_step)

  def LearningRate(self, step):
    p = self.params
    lrs = self.lr_schedule.Value(step)
    lrs.set_shape([])
    self._AddEvalMetric('lr_schedule', lrs, tf.constant(1.0))
    return p.learning_rate * lrs

  def Apply(self, loss, vmap, gradient_mask=None, gradient_adjuster=None):
    """Computes updates on 'vmap' to optimize 'loss'.

    TODO(rpang): explore merging gradient_mask and gradient_adjuster.

    Args:
      loss: A scalar Tensor.
      vmap: A `.NestedMap` object containing variables to optimize.
      gradient_mask: if not None, a dict mapping variable names to a 0/1 scalar.
      gradient_adjuster: if not None, a function that mutates a given var_grads.

    Returns:
      (op, eval_metrics), where op is a tf.Operation to update variables.
    """
    # We apply gradients outside the name_scope to maintain backwards
    # compatibility on variables created by self.optimizer.Apply().
    p = self.params

    vmap = self.GetTrainableVariables(vmap)

    for v in vmap.Flatten():
      tf.logging.info('%s: bprop variable: %s', p.name, v.name)

    # Compute gradients.
    var_grads = self.optimizer.ComputeGradients(
        loss,
        vmap,
        p.grad_aggregation_method,
        p.colocate_gradients_with_ops,
        p.gate_gradients,
        compute_gradients_fn=None,
        skip_zero_gradients=p.skip_zero_gradients)

    var_grads, stats = self.AdjustGradients(
        var_grads,
        gradient_mask=gradient_mask,
        gradient_adjuster=gradient_adjuster)
    self._var_grads = var_grads

    assert self.theta.global_step is not None, self.theta
    lr = self.LearningRate(self.theta.global_step)

    var_update_op = self.optimizer.Apply(lr, var_grads)
    return var_update_op, stats

  def AdjustGradients(self,
                      var_grads,
                      gradient_mask=None,
                      gradient_adjuster=None):
    """Adjusts gradients according to learner params.

    Args:
      var_grads: a `.NestedMap` whose values are (var, grad) pairs.
      gradient_mask: if not None, a dict mapping variable names to a 0/1 scalar.
      gradient_adjuster: if not None, a function that mutates a given var_grads.

    Returns:
      (var_grads, eval_metrics), where var_grads is a `.NestedMap` whose values
      (var, grad) pairs representing adjusted gradients.
    """
    p = self.params
    # L2 regularizer.
    if p.l2_regularizer_weight is not None:
      l2_loss, var_grads = py_utils.AdjustGradientsWithLpLoss(
          var_grads, p.l2_regularizer_weight, p=2.0)
      self._AddEvalMetric('l2_loss', l2_loss, tf.constant(1.0))

    # L1 regularizer.
    if p.l1_regularizer_weight is not None:
      l1_loss, var_grads = py_utils.AdjustGradientsWithLpLoss(
          var_grads, p.l1_regularizer_weight, p=1.0)
      self._AddEvalMetric('l1_loss', l1_loss, tf.constant(1.0))

    # Mask gradients only if the mask is set.
    if gradient_mask:
      var_grads = py_utils.MaskGradients(var_grads, gradient_mask)

    # Scale gradients, e.g., gradient clipping.
    if p.scale_gradients:
      scaled_vars = self.ScaleGradients(
          var_grads, gradient_adjuster=gradient_adjuster)
      var_grads = scaled_vars.final_var_grads

    # Histogram summary.
    summary_utils.CollectVarHistogram(var_grads)
    return var_grads, self._eval_metrics

  def _GetGlobalGradScale(self, all_grad_norm, has_nan_or_inf):
    """Returns a scaling factor for all gradients according to their norm.

    In case there are NaN or Inf values the function will return 0.0.

    Args:
      all_grad_norm: A scalar represeting the total norm of all vars.
      has_nan_or_inf: A scalar of 0 or 1, indicating whether there is any NaN or
        Inf in input gradients.

    Returns:
      The gradient scale. 0 if gradient updates should be skipped for the step.
    """
    p = self.params
    # Computes gradient's scale.
    grad_scale = tf.constant(1.0)
    if p.clip_gradient_norm_to_value:
      # If all_grad_norm > p.clip_gradient_norm_to_value, scales
      # all_grads so that the norm is 1.0.
      grad_scale = tf.minimum(1.0,
                              p.clip_gradient_norm_to_value / all_grad_norm)

    if p.grad_norm_to_clip_to_zero:
      # If all_grad_norm > p.grad_norm_to_clip_to_zero, treats
      # grad_scale as 0. This way, we ignore this step.
      grad_scale *= tf.cast(all_grad_norm < p.grad_norm_to_clip_to_zero,
                            p.dtype)

    if p.grad_norm_tracker:
      grad_scale *= self.grad_norm_tracker.FPropDefaultTheta(
          all_grad_norm, has_nan_or_inf)

    # Force grad_scale to be 0 if there is any NaN or Inf in gradients.
    grad_scale = tf.where(has_nan_or_inf, 0.0, grad_scale)

    return grad_scale

  def ScaleGradients(self, var_grads, gradient_adjuster=None):
    """Scales gradients according to training params.

    Args:
      var_grads: a `.NestedMap` whose values are (var, grad) pairs.
      gradient_adjuster: if not None, a function that mutates a given var_grads.

    Returns:
      A `.NestedMap` containing

      - final_var_grads: a `.NestedMap` whose values are (var, grad) pairs,
        where gradients have already been scaled.
      - grad_scale: the gradient scale. 0 if gradient updates should be skipped
        for the step. (Optional, only returned in case global norm clipping is
        used.)
    """
    p = self.params

    # Computes gradients' norm and adds their summaries. Note that all_grad_norm
    # may be nan, which may cause grad_scale to be nan.
    for name, vg in var_grads.FlattenItems():
      summary_utils.AddNormSummary(
          py_utils.SanitizeScopeKey(name) + '/' + p.name, vg)
    flatten = py_utils.Flatten(var_grads)
    all_grad_norm = tf.sqrt(py_utils.SumSquared([g for (_, g) in flatten]))
    all_var_norm = tf.sqrt(py_utils.SumSquared([v for (v, _) in flatten]))
    grad_norm_is_nan_or_inf = tf.math.logical_or(
        tf.math.is_nan(all_grad_norm), tf.math.is_inf(all_grad_norm))

    # Optional gradient adjustment. Note that this happens after computing
    # all_grad_norm.
    if gradient_adjuster is not None:
      tf.logging.info('gradient_adjuster=%s', gradient_adjuster)
      var_grads = gradient_adjuster(var_grads)

    # Handles NaN/Inf gradients.
    has_nan_or_inf = py_utils.HasNanOrInfGradient(var_grads)
    # Grad norm can still be inf even if none of the individual grad is inf.
    has_nan_or_inf = tf.math.logical_or(has_nan_or_inf, grad_norm_is_nan_or_inf)
    self._AddEvalMetric('has_nan_or_inf', has_nan_or_inf, tf.constant(1.0))

    return_values = py_utils.NestedMap()
    if p.clip_gradient_single_norm_to_value:
      # Currently using both types of clipping simultaneously is unsupported.
      if p.clip_gradient_norm_to_value:
        raise ValueError('Cannot use clip_gradient_single_norm_to_value=%f and '
                         'clip_gradient_norm_to_value=%f.' %
                         (p.clip_gradient_single_norm_to_value,
                          p.clip_gradient_norm_to_value))
      final_var_grads = py_utils.ApplyGradNormClipping(
          var_grads, p.clip_gradient_single_norm_to_value)

    else:
      grad_scale = self._GetGlobalGradScale(all_grad_norm, has_nan_or_inf)
      self._AddEvalMetric('grad_norm/all', all_grad_norm, tf.constant(1.0))
      self._AddEvalMetric('var_norm/all', all_var_norm, tf.constant(1.0))
      self._AddEvalMetric('grad_scale_all', grad_scale, tf.constant(1.0))
      final_var_grads = py_utils.ApplyGradMultiplier(var_grads, grad_scale)
      return_values.grad_scale = grad_scale

    return_values.final_var_grads = final_var_grads
    return return_values

  def _AddEvalMetric(self, key, value, weight):
    self._eval_metrics[key] = (value, weight)

_LEGACY_LEARNER_PARAMS = [
    'bprop_variable_filter',
    'bprop_variable_exclusion',
    'clip_gradient_norm_to_value',
    'clip_gradient_single_norm_to_value',
    'colocate_gradients_with_ops',
    'gate_gradients',
    'scale_gradients',
    'grad_aggregation_method',
    'grad_norm_to_clip_to_zero',
    'grad_norm_tracker',
    'l1_regularizer_weight',
    'l2_regularizer_weight',
    'learning_rate',
    'lr_schedule',
    'optimizer',
]


def ExtractLearnerFromLegacyParams(tp, cls=Learner):
  """Extracts legacy learner params from 'tp' to a Learner params.

  Args:
    tp: BaseTask training params (p.train). Its legacy params will be cleared to
      be None after the conversion.
    cls: Learner class where we set the params.

  Returns:
    A params for Learner.
  """
  lp = cls.Params()
  lp.name = 'loss'
  for k, v in tp.IterParams():
    if k not in _LEGACY_LEARNER_PARAMS:
      tf.logging.info(
          'Ignoring legacy param %s=%s for optimization program', k, v)
      continue
    setattr(lp, k, v)
    setattr(tp, k, None)
  for line in lp.ToText().split('\n'):
    tf.logging.info('Learner params: %s', line)
  return lp
