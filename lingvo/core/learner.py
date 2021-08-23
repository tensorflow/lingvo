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
"""A learner optimizes a subset of variables according to a loss.

It consists of a learning rate schedule, an optimizer, and gradient clipping
mechanisms. A BaseTask can have multiple learners, each optimizing a (usually
disjoint) subset of variables.
"""

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
    p = super().Params()
    p.Define(
        'l2_regularizer_weight', None,
        'If not None, L2 regularization to apply to the weights. '
        'Otherwise, disable L2 regularization.')
    p.Define(
        'loss_name', None, 'Name(s) of the loss(es) this learner to optimize. '
        'If not set, use learner name directly. '
        'If given as a list, the gradients will be combined via a '
        'GradientCombiner created from p.gradient_combiner, which must be '
        'specified as well.')
    p.Define(
        'gradient_combiner', None,
        'Params of a gradient_combiner.GradientCombiner used to combine '
        'gradients from multiple losses.')
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
        'similar to the behaviour of tf.clip_by_norm. Note this is mutually '
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
    p.Define(
        'learner_use_variable_scope', True,
        'Create children of learner in tf.variable_scope. This may need '
        'to be set to False for compatibility with the existing '
        'checkpoints trained from legacy code. New models should always '
        'set this to True.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    self._var_grads = None
    self._eval_metrics = {}

    # Don't create redundant variables in inference.
    is_training = not (self.do_eval or p.is_inference)
    if p.grad_norm_tracker and is_training:
      self.CreateChild('grad_norm_tracker', p.grad_norm_tracker)

    # TODO(b/184208049): don't create optimizer and lr_schedule in inference.
    self.CreateChild('optimizer', p.optimizer)
    self.CreateChild('lr_schedule', p.lr_schedule)
    if isinstance(p.loss_name, (list, tuple)):
      assert p.gradient_combiner
      self.CreateChild('gradient_combiner', p.gradient_combiner)
    else:
      assert p.gradient_combiner is None

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    p = self.params
    if not p.learner_use_variable_scope:
      # Note: multi learners fail in the legacy mode due to ValueError.
      # b/184208049
      if 'grad_norm_tracker' in self.children:
        self.grad_norm_tracker.InstantiateVariables()
      self.lr_schedule.InstantiateVariables()
      self.optimizer.InstantiateVariables()
    super()._CreateChildrenVariables()

  def GetVarGrads(self):
    return self._var_grads

  def GetTrainableVariables(self, vmap):
    p = self.params
    return py_utils.GetTrainableVariables(p.name, p.bprop_variable_filter,
                                          p.bprop_variable_exclusion, vmap)

  def ApplyPostTrainingLoop(self):
    """Applies any computation to run after each tpu trainining loop.

    Returns:
      Ops to run after training loop ends.
    """
    return self.optimizer.ApplyPostTrainingLoop()

  def LearningRate(self):
    p = self.params
    lrs = self.lr_schedule.Value()
    lrs.set_shape([])
    self._AddEvalMetric('lr_schedule', lrs, tf.constant(1.0))
    return p.learning_rate * lrs

  def Apply(self, metrics, vmap, gradient_mask=None, gradient_adjuster=None):
    """Computes updates on 'vmap' to optimize 'loss'.

    TODO(rpang): explore merging gradient_mask and gradient_adjuster.

    Args:
      metrics: A Dict[str, (value, weight)], from which loss can be extracted
        according to p.loss_name.
      vmap: A `.NestedMap` object containing variables to optimize.
      gradient_mask: if not None, a dict mapping variable names to a 0/1 scalar.
      gradient_adjuster: if not None, a function that mutates a given var_grads.

    Returns:
      (losses, op, eval_metrics), where
        - losses is a list of scalar tensors;
        - op is a tf.Operation to update variables;
        - eval_metrics is a Dict[str, (value, weight)], where each value/weight
          is a scalar tensor.
    """
    # We apply gradients outside the name_scope to maintain backwards
    # compatibility on variables created by self.optimizer.Apply().
    losses, var_grads, eval_metrics = self._ComputeLossesAndGradients(
        metrics, vmap)
    if 'tpu_embedding_var_grads' in var_grads:
      tpu_embedding_var_grads = var_grads.tpu_embedding_var_grads
      del var_grads.tpu_embedding_var_grads

      tpu_embedding_collection = py_utils.GetTpuEmbeddingGraphCollection()[0]
      assert tpu_embedding_collection
      tpu_emb_update_op, stats = tpu_embedding_collection.ApplyGradients(
          py_utils.GetTaskCallScope(),
          tpu_embedding_var_grads.Transform(lambda var_grad: var_grad.grad))
      eval_metrics.update(stats)
    else:
      tpu_emb_update_op = tf.no_op()

    assert py_utils.GetGlobalStep() is not None
    lr = self.LearningRate()

    var_grads, stats = self.AdjustGradients(
        var_grads,
        gradient_mask=gradient_mask,
        gradient_adjuster=gradient_adjuster)
    eval_metrics.update(stats)
    self._var_grads = var_grads

    eval_metrics['learning_rate'] = (tf.convert_to_tensor(lr),
                                     tf.convert_to_tensor(1.))

    var_update_op = tf.group(
        [tpu_emb_update_op,
         self.optimizer.Apply(lr, var_grads)])
    return losses, var_update_op, eval_metrics

  def ComputeActivationGradients(self, activations, activations_grad, vmap):
    p = self.params
    vmap = self.GetTrainableVariables(vmap)

    for v in vmap.Flatten():
      tf.logging.info('%s: bprop variable: %s', p.name, v.name)
    return self.optimizer.ComputeGradients(
        activations,
        vmap,
        p.grad_aggregation_method,
        p.colocate_gradients_with_ops,
        p.gate_gradients,
        compute_gradients_fn=self._CustomComputeGradientsFn(),
        skip_zero_gradients=p.skip_zero_gradients,
        skip_none_gradients=False,
        activations_grad=activations_grad,
        is_activations=True)

  def ComputeLosses(self, metrics):
    p = self.params

    def _Loss(metric_name):
      """Returns (loss, var_grads) computed from metrics[metric_name]."""
      metric = metrics.get(metric_name, None)
      if metric is None:
        raise ValueError('Loss %s not found in metrics %s' %
                         (metric_name, list(metrics.keys())))
      return metric

    loss_name = p.loss_name or p.name
    losses = []
    if isinstance(loss_name, (list, tuple)):
      for metric_name in loss_name:
        loss_metric = _Loss(metric_name)
        losses.append(loss_metric[0])
    else:
      loss_metric = _Loss(loss_name)
      losses.append(loss_metric[0])

    return losses

  def _CustomComputeGradientsFn(self):
    """Returns the compute_gradients_fn to use for py_utils.ComputeGradients."""
    return None  # use the default function

  def _ComputeLossesAndGradients(self, metrics, vmap):
    p = self.params
    vmap = self.GetTrainableVariables(vmap)

    # Get tpu embedding activations to compute the gradients for.
    tpu_embedding_activations = py_utils.NestedMap()
    tpu_embedding_graph_collection = py_utils.GetTpuEmbeddingGraphCollection()
    if tpu_embedding_graph_collection:
      tpu_embedding_collection = tpu_embedding_graph_collection[0]
      task_call_scope = py_utils.GetTaskCallScope()
      tpu_embedding_activations = py_utils.NestedMap(
          tpu_embedding_collection.GetActivations(task_call_scope) or {})
      # It's possible that task_call_scope is None and its mode is not set in
      # tpu_embedding_collection (e.g. in unit test), but if the activation is
      # not empty, the mode must have been set.
      if tpu_embedding_activations and (
          tpu_embedding_collection.ShouldStopGradient(task_call_scope)):
        tpu_embedding_activations = py_utils.NestedMap()

    for v in vmap.Flatten():
      tf.logging.info('%s: bprop variable: %s', p.name, v.name)

    def LossAndGradients(metric_name):
      """Returns (loss, var_grads) computed from metrics[metric_name]."""
      metric = metrics.get(metric_name, None)
      if metric is None:
        raise ValueError('Loss %s not found in metrics %s' %
                         (metric_name, list(metrics.keys())))
      # TODO(b/154785713): pass (loss, loss_weight) to ComputeGradients().
      loss = metric[0]
      return metric, self.optimizer.ComputeGradients(
          loss,
          vmap,
          p.grad_aggregation_method,
          p.colocate_gradients_with_ops,
          p.gate_gradients,
          compute_gradients_fn=self._CustomComputeGradientsFn(),
          skip_zero_gradients=p.skip_zero_gradients,
          skip_none_gradients=False,
          tpu_embedding_activations=tpu_embedding_activations)

    loss_name = p.loss_name or p.name
    losses = []
    eval_metrics = {}
    if isinstance(loss_name, (list, tuple)):
      assert not tpu_embedding_activations, (
          'TPU embedding does not support multiple loss currently.')
      losses_and_grads = {}
      variables = None
      for metric_name in loss_name:
        loss_metric, var_grads = LossAndGradients(metric_name)
        losses_and_grads[metric_name] = py_utils.NestedMap(
            loss_metric=loss_metric,
            grads=tf.nest.map_structure(lambda vg: vg.grad, var_grads))
        current_vars = tf.nest.map_structure(lambda vg: vg.var, var_grads)
        if variables is None:
          variables = current_vars
        else:
          tf.nest.assert_same_structure(variables, current_vars)
        losses.append(loss_metric[0])

      grads, eval_metrics = self.gradient_combiner.Combine(
          variables, losses_and_grads)
      var_grads = tf.nest.map_structure(
          lambda v, g: py_utils.VarGrad(var=v, grad=g), variables, grads)
    else:
      loss_metric, var_grads = LossAndGradients(loss_name)
      losses.append(loss_metric[0])

    return losses, py_utils.SkipNoneGradients(var_grads), eval_metrics

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
    grad_scale = tf.constant(1.0, all_grad_norm.dtype)
    if p.clip_gradient_norm_to_value:
      # If all_grad_norm > p.clip_gradient_norm_to_value, scales
      # all_grads so that the norm is 1.0.
      grad_scale = tf.minimum(
          tf.constant(1.0, all_grad_norm.dtype),
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
    grad_scale = tf.where(has_nan_or_inf, tf.constant(0.0, grad_scale.dtype),
                          grad_scale)

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
    self._AddEvalMetric('grad_norm_is_nan_or_inf', grad_norm_is_nan_or_inf,
                        tf.constant(1.0))

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
      # grad_norm/all is both a eval metric(collected by trainer) and a summary
      # (collected by controller).
      summary_utils.scalar(f'grad_norm/all/{p.name}', all_grad_norm)
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
    'learner_use_variable_scope',
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
