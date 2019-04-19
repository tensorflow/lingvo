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
"""Base model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
from six.moves import range
import tensorflow as tf

from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import build_data
from lingvo.core import cluster_factory
from lingvo.core import early_stop
from lingvo.core import hyperparams
from lingvo.core import lr_schedule
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import summary_utils
from lingvo.core import task_scheduler


def CreateTaskGlobalStep(task_name):
  """Create if needed and return the global_step."""
  with tf.name_scope(None), tf.variable_scope(py_utils.global_variable_scope):
    graph_collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'TASK_GLOBAL_STEP']
    _, v = py_utils.CreateVariable(
        name=task_name + '_global_step',
        params=py_utils.WeightParams([], py_utils.WeightInit.Constant(0),
                                     tf.int64),
        trainable=False,
        collections=graph_collections)
    summary_utils.scalar(v.name, v)
    return v


class StatsCounter(object):
  """A single counter in TF."""

  def __init__(self, name):
    self._name = name
    _, self._var = py_utils.CreateVariable(
        name=name,
        params=py_utils.WeightParams([], py_utils.WeightInit.Constant(0),
                                     tf.int64),
        trainable=False)
    self._value = self._var.value() + 0  # Makes a copy.

  def Value(self):
    """Returns the current counter value."""
    return self._value

  def IncBy(self, params, delta):
    """Increment the counter by delta and return the new value."""
    # NOTE: We must ensure _value is computed (_var + 0) before
    # updating _var with delta.
    delta = tf.to_int64(delta)
    with tf.control_dependencies([self._value]):
      summary_utils.scalar(self._name, self._value)
      return tf.identity(tf.assign_add(self._var, delta))


class BaseTask(base_layer.BaseLayer):
  """A single encoder/decoder task.

  One task usually consists of one InputGenerator, one train_op,
  a list of eval_metrics, etc.
  """

  @classmethod
  def Params(cls):
    p = super(BaseTask, cls).Params()
    p.Define('input', None, 'Input generator Params.')
    p.Define('encoder', None, 'Encoder Params.')
    p.Define('online_encoder', None, 'Online Encoder Params.')
    p.Define('decoder', None, 'Decoder Params.')
    p.Define('train', hyperparams.Params(),
             'Params to control how this task should be trained.')

    tp = p.train
    tp.Define(
        'start_up_delay_steps', 200, 'i-th replica starts training after '
        'i*(i+1)/2*start_up_delay_steps steps')
    tp.Define('max_steps', 4 * 10**6, 'Maximum number of training steps.')
    tp.Define('tpu_steps_per_loop', 100, 'The number of training steps per '
              'training loop for TPUs.')
    tp.Define(
        'vn_start_step', 200000000,
        'Step starting from which variational noise is added to '
        'params values during training.')
    tp.Define('vn_std', 0.0, 'Std of the variational noise.')
    tp.Define(
        'l2_regularizer_weight', None,
        'If not None, L2 regularization to apply to the weights. '
        'Otherwise, disable L2 regularization.')
    tp.Define(
        'l1_regularizer_weight', None,
        'If not None, L1 regularization to apply to the weights. '
        'Otherwise, disable L1 regularization.')
    tp.Define('learning_rate', 0.0, 'learning rate to use.')
    tp.Define(
        'clip_gradient_norm_to_value', 0.0,
        'Clip gradient by global norm to this value. This is similar to '
        'the bahaviour of tf.clip_by_global_norm, if you are looking for '
        'tf.clip_by_norm refer to clip_gradient_single_norm_to_value. Note '
        'these are mutually exclusive.')
    tp.Define(
        'clip_gradient_single_norm_to_value', 0.0,
        'Clip gradient by single tensor norm to this value. This is '
        'similar to the bahaviour of tf.clip_by_norm. Note this is mutually '
        'exlusive to using clip_gradient_norm_to_value.')
    tp.Define('grad_norm_to_clip_to_zero', 0.0,
              'Clip gradient to 0 if its norm exceeds this value.')
    tp.Define('grad_norm_tracker', None, 'Params for GradNormTracker.')
    tp.Define('optimizer', optimizer.Adam.Params(), 'Params for the optimizer.')
    tp.Define('lr_schedule',
              lr_schedule.ContinuousLearningRateSchedule.Params(),
              'Learning rate decay schedule.')
    tp.Define('early_stop', early_stop.EarlyStop.Params(),
              'Early stopping based on dev-set performance.')
    tp.Define(
        'ema_decay', 0.0,
        'If > 0, enable ExponentialMovingAverage during training '
        'with the give decay. '
        'Must be < 1. Disabled if <= 0.')
    tp.Define(
        'bprop_variable_filter', None,
        'If set, only backprop variables whose names partially match '
        'this regexp (re.search).')
    tp.Define(
        'init_from_checkpoint_rules', {},
        'If not None, a dictionary with keys corresponding to a checkpoint '
        'path and values corresponding to variable loading rules is expected. '
        'Each key is expected to be a path to a checkpoint from which to '
        'initialize part of the model. Variables are only loaded from this '
        'path during initialization and will override values provided by '
        'initialization.'
        'The corresponding values (loading_rules) are expected to be a tuple '
        'consisting of two list: loading rules, and ignore rules, respectively.'
        'The first list (loading rules) contains the list of variables '
        'which should be initialized from the checkpoint: each element in the '
        'list is a pair of strings. The first element is a regex and the '
        'second is a python format string. If a variable in the model matches '
        'a regex, we rename using the format string to determine the '
        'corresponding var in the checkpoint. Note that, it is an error if a '
        'model variable matches multiple loading rules, for the same '
        'checkpoint or across checkpoints.'
        'The second list (ignore rules) is a list of regexes which specify '
        'variables in the model which should not be initialized using the '
        'loading rules. Thus, if a variable in the model to be trained matches '
        'one of the rules in the loading rules, as well as one of the regular '
        'expressions in the ignore rules, the variable will not be initialized '
        'from the checkpoint, but will instead be initialized from the '
        'variable initalizer defined in the graph.'
        'Examples:'
        '{"checkpoint_path": ([("(.*)", "%s")], [])} will initialize all the '
        'model parameters from the checkpoint_path.')
    tp.Define(
        'pruning_hparams_dict', None, 'Pruning related hyperparameters. A dict '
        'with hyperparameter: value pairs. See tf.contrib.model_pruning.')
    tp.Define('save_interval_seconds', 60 * 10,
              'Generates a checkpoint roughly once every this many seconds.')
    tp.Define('summary_interval_steps', 100,
              'Generates a checkpoint roughly once every this many steps.')
    tp.Define(
        'grad_aggregation_method', tf.AggregationMethod.EXPERIMENTAL_TREE,
        'Specifies the method used to combine gradient terms. Accepted '
        'values are constants defined in the class AggregationMethod.')
    tp.Define(
        'gate_gradients', False,
        'If True, add a tuple around the gradients returned for an '
        'operations. This avoids some race conditions.')
    tp.Define('colocate_gradients_with_ops', True,
              'If True, try colocating gradients with the corresponding op.')
    p.Define('eval', hyperparams.Params(),
             'Params to control how this task should be evaled.')
    ep = p.eval
    ep.Define(
        'samples_per_summary', 1000,
        'If > 0, generates one summary after this many samples, at most. '
        'If == 0 or the dataset has fewer examples, evaluate the whole set.')
    ep.Define(
        'decoder_samples_per_summary', 0,
        'If > 0, each decoder summary will contain at most this many samples. '
        'If == 0, defaults to `samples_per_summary` for '
        'backwards compatibility.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    assert issubclass(params.cls, BaseTask)
    # Ensure global_step exists before calling super.
    py_utils.GetOrCreateGlobalStepVar()
    super(BaseTask, self).__init__(params)

    p = self.params

    if p.input:
      # TODO(zhifengc): Consider a simpler way to ensure the input
      # generator stops after one epoch.
      if p.is_eval and p.eval:
        seq_inp = issubclass(p.input.cls,
                             base_input_generator.BaseInputGeneratorFromFiles)
        if p.input.num_samples == 0:
          # Dataset size is unknown. Computes eval summary based on num_samples.
          assert p.eval.samples_per_summary > 0
        elif (p.eval.samples_per_summary == 0) or (p.input.num_samples <
                                                   p.eval.samples_per_summary):
          # If we know the dataset size and we want to evaluate the full
          # set, we need to coordinate the input generator to flush out
          # all samples so the evaler and decoder compute metrics on the
          # whole set for each summary step.
          if seq_inp:
            p.input.flush_every_n = p.input.num_samples
          p.eval.samples_per_summary = p.input.num_samples
        if seq_inp and p.input.num_batcher_threads > 1:
          tf.logging.warning('input.num_batcher_threads > 1 inside eval mode.  '
                             'The input generator may not iterate over exactly '
                             'one epoch per run')

      with tf.device(
          self.cluster.input_device), py_utils.outside_all_rewrites():
        self.CreateChild('input', p.input)

    self._var_grads = None
    self._encoder = None
    self._online_encoder = None
    self._decoder = None

    self._total_examples = None
    self._total_nans_and_infs = None
    self._loss = None
    self._num_predictions = None
    self._train_op = None
    self._eval_metrics = {}
    self._per_example = {}
    self._trainer_verbose_tensors = {}

    # Create the gradient mask,
    self._per_input_gradient_mask = None
    task_global_step_list = tf.get_collection('TASK_GLOBAL_STEP',
                                              '^%s_global_step' % p.name)
    if len(task_global_step_list) > 1:
      raise ValueError('Found multiple task_global_step for task %s' % p.name)
    self._global_step_var = (
        task_global_step_list[0] if len(task_global_step_list) == 1 else
        py_utils.GetOrCreateGlobalStepVar())
    self._global_step = tf.identity(
        self._global_step_var, name='global_step_tensor')
    tp = p.train
    # p.train can be None if this task is the teacher/student task in a
    # DistillationTask.
    if tp and self.cluster.job in ('worker', 'trainer', 'trainer_client',
                                   'controller'):
      if tp.grad_norm_tracker:
        with tf.variable_scope(p.name):
          self.CreateChild('grad_norm_tracker', tp.grad_norm_tracker)

      self.CreateChild('lr_schedule', tp.lr_schedule)
      self.CreateChild('optimizer', tp.optimizer)
    self._UpdateVnConfig()

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for `input_batch`.

    The output can be in the form of probablistic distributions, e.g., softmax
    logits for discrete outputs, mixture of logistics for continuous values, or
    regression values.

    For training/evaluation, the output will be used for computing loss and
    gradient updates, including comparing predicted distributions between
    teacher and student for distillation. During inference the output can be
    used to compute final outputs, perhaps with sampling.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      Predictions, either a single Tensor, a `.NestedMap`, or a namedtuple.
    """
    raise NotImplementedError('Abstract method')

  def ComputeLoss(self, theta, input_batch, predictions):
    """Computes loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors to this tower.
      predictions: The output of `ComputePredictions`.

    Returns:
      Two dicts:
      A dict containing str keys and (metric, weight) pairs as values, where
      one of the keys is expected to be 'loss'.
      A dict containing arbitrary tensors describing something about each
      training example, where the first dimension of each tensor is the batch
      index.
    """
    raise NotImplementedError('Abstract method')

  def FilterPerExampleTensors(self, per_example):
    """Return the per-example tensors ProcessFPropResults needs.

    By default we don't send any per-example tensors to ProcessFPropResults
    because some may be expensive to compute. Implement this method to let
    some of them pass through.

    Args:
      per_example: A dict of tensors returned as per-example tensors from FProp.

    Returns:
      A dict containing a subset of the key/value pairs in per_example.
    """
    return {}

  def ProcessFPropResults(self, sess, global_step, metrics, per_example):
    """Called once for each train loop.

    BaseModel.ProcessFPropResults is also called on each loop, so you
    can put your implementation wherever it is most convenient for you.

    Args:
      sess: a session.
      global_step: approximate number of model training steps.
      metrics: the metrics dict returned by FPropTower.
      per_example: the per_example dict returned by FPropTower.
    """
    pass

  def FPropTower(self, theta, input_batch):
    """Forward propagation through one tower of the model.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      Two dicts:
      A dict containing str keys and (metric, weight) pairs as values, where
      one of the keys is expected to be 'loss'.
      A dict containing arbitrary tensors describing something about each
      training example, where the first dimension of each tensor is the batch
      index.
    """
    predicted = self.ComputePredictions(theta, input_batch)
    return self.ComputeLoss(theta, input_batch, predicted)

  def FProp(self, theta, input_batch):
    """Forward propagation.

    This default `FProp` implementation here supports batch splitting in
    synchronous and asynchronous training when sub-classes implement
    `FPropTower`.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
        spiltting is used, a list of `NestedMap`, one for each split.

    Returns:
      Two dicts:
      A dict containing str keys and (metric, weight) pairs as values, where
      one of the keys is expected to be 'loss'.
      A dict containing arbitrary tensors describing something about each
      training example, where the first dimension of each tensor is the batch
      index.
    """
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      # Always reset step seed at the start of a new global_step.
      py_utils.ResetStepSeed()
      if py_utils.use_tpu():
        metrics, per_example = self._FPropTpu(theta, input_batch)
      else:
        metrics, per_example = self._FPropSplitInputBatch(theta, input_batch)
      self._FPropResult(metrics, per_example)
    return metrics, per_example

  def _FPropTpu(self, theta, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      with tf.name_scope('tower_0_0'):
        metrics, per_example = self.FPropTower(theta, input_batch)
        metrics = py_utils.WeightedAvgOfMetrics([metrics])
    return metrics, per_example

  def _FPropSplitInputBatch(self, theta, input_batch):
    """Splits the input batch on the input device."""
    cluster = self.cluster
    num_splits = cluster.num_splits_per_client

    if not isinstance(input_batch, list):
      input_batch = [input_batch]

    assert len(input_batch) == num_splits, (len(input_batch), num_splits)

    # dev_list_per_replica[i][j] is the i-th worker's j-th device.
    dev_list_per_replica = cluster.available_devices.tolist()

    # Asserts invariant of the total number of splits w.r.t.,
    # splits per worker.
    splits_per_replica = cluster.num_splits_per_replica
    assert num_splits == splits_per_replica * len(dev_list_per_replica), (
        num_splits, splits_per_replica, len(dev_list_per_replica))

    all_metrics = []
    all_per_example_tensors = []
    for w_id, w_devs in enumerate(dev_list_per_replica):
      # Make local copy of the vars, shard on devices for this worker.
      theta_local = py_utils.CreateLocalTheta(
          theta, w_devs, label='worker %d' % w_id)

      for s_id in range(splits_per_replica):
        # s_id-th split for the w_id-th worker.
        split_id = splits_per_replica * w_id + s_id
        with py_utils.ModelSplit(split_id):
          with tf.device(cluster.WorkerDeviceInModelSplit(0)):
            with tf.name_scope('tower_%d_%d' % (w_id, s_id)):
              batch = self.input_generator.PreprocessInputBatch(
                  input_batch[split_id])
              metrics, per_example = self.FPropTower(theta_local, batch)
        all_metrics.append(metrics)
        all_per_example_tensors.append(per_example)

    return py_utils.WeightedAvgOfMetrics(
        all_metrics), py_utils.ConcatPerExampleTensors(all_per_example_tensors)

  def _FPropResult(self, metrics, per_example):
    # Adds stats about the input batch.
    metrics['num_samples_in_batch'] = (tf.convert_to_tensor(
        self.input_generator.GlobalBatchSize()), tf.constant(1.0))
    # Generates summaries.
    for name, (value, weight) in six.iteritems(metrics):
      self.AddEvalMetric(name, value, weight)
    per_example = self.FilterPerExampleTensors(per_example)
    for name, value in six.iteritems(per_example):
      self.AddPerExampleTensor(name, value)
    # Loss.
    self._loss, self._num_predictions = metrics['loss']
    self._loss = py_utils.CheckNumerics(self._loss)
    self._metrics = metrics
    summary_utils.scalar('num_predictions', self._num_predictions)

  def GetInputBatch(self):
    """Returns input batch from input_generator."""
    if py_utils.use_tpu():
      return self.input_generator.CreateTpuFeeds()
    else:
      cluster = self.cluster
      num_splits = cluster.num_splits_per_client
      with tf.device(cluster.input_device):
        return self.input_generator.SplitInputBatch(num_splits)

  def FPropDefaultTheta(self, input_batch=None):
    """Calls `FProp` with this layer's parameters."""
    if input_batch is None:
      input_batch = self.GetInputBatch()
    return self.FProp(self.theta, input_batch)

  def GetVarGrads(self):
    return self._var_grads

  def AdjustGradients(self, vars_gradients):
    """Allow for custom gradient manipulation prior to clipping."""
    return vars_gradients

  def BProp(self):
    """Constructs the backward graph."""
    p = self.params
    vs = self.vars
    bprop_variable_filters = self.input_generator.GetBpropVariableFilters()
    # Only compute the mask if the variable filters are not empty.
    if bprop_variable_filters != [''] * len(bprop_variable_filters):
      self._ComputeGradientMask(bprop_variable_filters)
    if p.train.bprop_variable_filter:

      def VariableFilter(v):
        if re.search(p.train.bprop_variable_filter, v.name):
          return True
        tf.logging.info('bprop disabled by bprop_variable_filter: %s', v.name)
        return False

      vs = vs.Filter(VariableFilter)
      tf.logging.info('Filtered bprop variables: %s', vs)
    self._BPropForVariables(vs)

  def _ComputeGradientMask(self, bprop_variable_filters):
    """Compute gradient mask for each variable and bprop_variable_filters.

    Note that per_input_gradient_mask[var][i] will be 1 if var matches
    bprop_variable_filter[i], 0 otherwise.

    Args:
      bprop_variable_filters: A list of regex bprop_variable_filters for each
        file pattern.
    """
    self._per_input_gradient_mask = py_utils.NestedMap()
    all_vars = set(self.vars.Flatten())
    for var in all_vars:
      self._per_input_gradient_mask[var.name] = (
          tf.zeros(len(bprop_variable_filters), dtype=tf.float32))
      for i in range(len(bprop_variable_filters)):
        if re.search(bprop_variable_filters[i], var.name):
          self._per_input_gradient_mask[var.name] += (
              tf.one_hot(i, len(bprop_variable_filters), dtype=tf.float32))

  def _HasNanOrInf(self, var_grads):
    """Returns a bool tensor to indicate if `var_grads` contains NaNs or Infs.

    Args:
      var_grads: A `.NestedMap` with (var, grad) tuple as the map value.

    Returns:
      A bool scalar tensor to indicate if the `var_grads` contains NaNs or Infs.
    """

    def HasNanOrInf(x):
      with tf.device(x.device):
        if x.dtype.is_complex:
          return tf.reduce_any(
              [HasNanOrInf(tf.real(x)),
               HasNanOrInf(tf.imag(x))])
        return tf.reduce_any(tf.logical_or(tf.is_nan(x), tf.is_inf(x)))

    return tf.reduce_any([(HasNanOrInf(g.values) if isinstance(
        g, tf.IndexedSlices) else HasNanOrInf(g))
                          for (_, g) in var_grads.Flatten()])

  def _GetGlobalGradScale(self, all_grad_norm, has_nan_or_inf):
    """Returns a scaling factor for all gradients according to their norm.

    In case there are NaN or Inf values the function will return 0.0.

    Args:
      all_grad_norm: A scalar represeting the total norm of all vars.
      has_nan_or_inf: A scalar of 0 or 1, indicating whether there is any NaN or
        Inf in input gradients.

    Returns:
      grad_scale: the gradient scale. 0 if gradient updates should be skipped
        for the step.
    """
    p = self.params
    tp = p.train
    # Computes gradient's scale.
    grad_scale = tf.constant(1.0)
    if tp.clip_gradient_norm_to_value:
      # If all_grad_norm > tp.clip_gradient_norm_to_value, scales
      # all_grads so that the norm is 1.0.
      grad_scale = tf.minimum(1.0,
                              tp.clip_gradient_norm_to_value / all_grad_norm)

    if tp.grad_norm_to_clip_to_zero:
      # If all_grad_norm > tp.grad_norm_to_clip_to_zero, treats
      # grad_scale as 0. This way, we ignore this step.
      grad_scale *= tf.cast(all_grad_norm < tp.grad_norm_to_clip_to_zero,
                            p.dtype)

    if tp.grad_norm_tracker:
      grad_scale *= self.grad_norm_tracker.FPropDefaultTheta(
          all_grad_norm, has_nan_or_inf)

    # Force grad_scale to be 0 if there is any NaN or Inf in gradients.
    grad_scale = tf.where(has_nan_or_inf, 0.0, grad_scale)

    return grad_scale

  def ScaleGradients(self, var_grads):
    """Scales gradients according to training params.

    Args:
      var_grads: a `.NestedMap` whose values are (var, grad) pairs.

    Returns:
      A `.NestedMap` containing:
      - has_nan_or_inf: a scalar of 0 or 1, indicating whether there is any NaN
        or Inf in input gradients.
      - final_var_grads: a `.NestedMap` whose values are (var, grad) pairs,
        where gradients have already been scaled.
      - grad_scale: the gradient scale. 0 if gradient updates should be skipped
        for the step. (Optional, only returned in case global norm clipping is
        used.)
    """
    p = self.params
    tp = p.train

    # Computes gradients' norm and adds their summaries. Note that all_grad_norm
    # may be nan, which may cause grad_scale to be nan.
    for name, vg in var_grads.FlattenItems():
      summary_utils.AddNormSummary(name, py_utils.NestedMap(s=vg))
    all_grad_norm = tf.sqrt(
        py_utils.SumSquared(
            [g for (_, g) in py_utils.NestedMap(child=var_grads).Flatten()]))
    all_var_norm = tf.sqrt(
        py_utils.SumSquared(
            [v for (v, _) in py_utils.NestedMap(child=var_grads).Flatten()]))
    grad_norm_is_nan_or_inf = tf.logical_or(
        tf.is_nan(all_grad_norm), tf.is_inf(all_grad_norm))

    # Optional gradient adjustment. Note that this happens after computing
    # all_grad_norm.
    var_grads = self.AdjustGradients(var_grads)

    # Handles NaN/Inf gradients.
    has_nan_or_inf = self._HasNanOrInf(var_grads)
    # Grad norm can still be inf even if none of the individual grad is inf.
    has_nan_or_inf = tf.logical_or(has_nan_or_inf, grad_norm_is_nan_or_inf)

    return_values = py_utils.NestedMap()
    if tp.clip_gradient_single_norm_to_value:
      # Currently using both types of clipping simultaneously is unsupported.
      if tp.clip_gradient_norm_to_value:
        raise ValueError('Cannot use clip_gradient_single_norm_to_value=%f and '
                         'clip_gradient_norm_to_value=%f.' %
                         (tp.clip_gradient_single_norm_to_value,
                          tp.clip_gradient_norm_to_value))
      final_var_grads = py_utils.ApplyGradNormCliping(
          var_grads, tp.clip_gradient_single_norm_to_value)

    else:
      grad_scale = self._GetGlobalGradScale(all_grad_norm, has_nan_or_inf)
      self.AddEvalMetric('grad_norm/all', all_grad_norm, tf.constant(1.0))
      self.AddEvalMetric('var_norm/all', all_var_norm, tf.constant(1.0))
      self.AddEvalMetric('grad_scale_all', grad_scale, tf.constant(1.0))
      final_var_grads = py_utils.ApplyGradMultiplier(var_grads, grad_scale)
      return_values.grad_scale = grad_scale

    return_values.has_nan_or_inf = has_nan_or_inf
    return_values.final_var_grads = final_var_grads
    return return_values

  def _BPropForVariables(self, vmap):
    """Constructs the backward graph for the given variables.

    Args:
      vmap: a `.NestedMap` of variables.
    """
    p = self.params
    tp = p.train

    # Compute gradients.
    self._var_grads = py_utils.ComputeGradients(
        self.loss, vmap, tp.grad_aggregation_method,
        tp.colocate_gradients_with_ops, tp.gate_gradients)

    # L2 regularizer.
    if tp.l2_regularizer_weight is not None:
      l2_loss, self._var_grads = py_utils.AdjustGradientsWithLpLoss(
          self._var_grads, tp.l2_regularizer_weight, p=2.0)
      summary_utils.scalar('l2_loss', l2_loss)

    # L1 regularizer.
    if tp.l1_regularizer_weight is not None:
      l1_loss, self._var_grads = py_utils.AdjustGradientsWithLpLoss(
          self._var_grads, tp.l1_regularizer_weight, p=1.0)
      summary_utils.scalar('l1_loss', l1_loss)

    # Mask gradients only if the mask is set.
    if self._per_input_gradient_mask:
      bprop_onehot = self.input_generator.GetInputSourceOneHot()
      self._var_grads = py_utils.MaskGradients(
          self._var_grads, self._per_input_gradient_mask, bprop_onehot)

    # Apply gradient clipping.
    scaled_vars = self.ScaleGradients(self._var_grads)
    has_nan_or_inf = scaled_vars.has_nan_or_inf
    self._var_grads = scaled_vars.final_var_grads

    # Histogram summary.
    summary_utils.CollectVarHistogram(self._var_grads)

    lrs = self.lr_schedule.Value(self.global_step)
    summary_utils.scalar('lr_schedule', lrs)
    lr = tp.learning_rate * lrs

    var_update_op = self.optimizer.Apply(lr, self._var_grads)

    relevant_bn_updates, _ = py_utils.FindRelevantBatchNormUpdates(
        self.loss, tf.get_collection(py_utils.BATCH_NORM_UPDATES))
    batch_norm_updates = tf.group(*relevant_bn_updates)

    # Update stats.
    stats_updates = tf.group(
        self.IncrementTotalSamples(),
        self.IncrementTotalNans(tf.to_int32(has_nan_or_inf)))

    # Post training step update.
    post_training_step_updates = self.PostTrainingStepUpdate(self.global_step)

    # Get the op to update the weight masks and thresholds
    mask_update_op = self._GetMaskUpdateOp()

    # TODO(rpang): try to structure _train_op as:
    #   tf.cond(skip_step, <only update skip stats>, <all updates>)
    # so that we skip all other updates when a step is skipped.
    train_ops = [
        var_update_op, batch_norm_updates, stats_updates,
        post_training_step_updates, mask_update_op
    ]
    with tf.control_dependencies(train_ops):
      true_global_step = py_utils.GetOrCreateGlobalStepVar()
      with tf.colocate_with(true_global_step):
        increment_global_steps = tf.assign_add(true_global_step, 1)
      if self._global_step_var != true_global_step:
        with tf.colocate_with(self._global_step_var):
          increment_global_steps = tf.group(
              increment_global_steps, tf.assign_add(self._global_step_var, 1))

    train_ops.append(increment_global_steps)
    self._train_op = tf.group(*train_ops, name='train')

  def ApplyExponentialMovingAverage(self, ema):
    """Wraps `self.train_op` with an op updating exponential moving average."""
    # We need to apply EMA to trainable and moving average variable of this
    # Task, not just bprop vars, so that we create a shadow
    # '/ExponentialMovingAverage' variable for every trainable and moving
    # average variable.
    all_vars = set(tf.trainable_variables()) | set(
        tf.moving_average_variables())
    all_vars &= set(self.vars.Flatten())
    for var in all_vars:
      tf.logging.debug('ApplyExponentialMovingAverage: %s', var.name)
    with tf.control_dependencies(
        [self._train_op]), tf.name_scope('moving_average'):
      self._train_op = ema.apply(all_vars)

  def Decode(self, input_batch):
    """Constructs the inference graph for eval decoding.

    Args:
      input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
        spiltting is used, a list of `NestedMap`, one for each split.

    Returns:
      a dict of Tensors as decoder output.
    """
    return {}

  def Inference(self):
    """Constructs the inference graph.

    Each subgraph represents a public API for a part of the graph which can
    be operated independently. By convention, the subgraph named 'default'
    should perform end to end inference via the input generator.

    Note that having distinct subgraphs (e.g. 'encoder', 'decoder') is
    not just a space optimization: when driving the graph externally in an
    online fashion, evaluation often needs to be broken into pieces. In this
    case, the graph will be constructed with only those pieces.

    Returns:
      An `inference_graph_pb2.InferenceGraph` message.
    """
    raise NotImplementedError('Abstract method')

  def CreateDecoderMetrics(self):
    """Creates a dict of decoder metrics for `PostProcessDecodeOut` to update.

    Returns a dict mapping from string keys to `.BaseMetric` objects.
    """
    pass

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out and updates contents of `decode_metrics_dict`.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to `.BaseMetric`
        object as created by `CreateDecoderMetrics`.

    Returns:
      output_key_value_pairs - a list of (key, value) pairs that can be saved
      (i.e. of type str, bytes, or unicode).
    """
    pass

  @property
  def loss(self):
    assert self._loss is not None, ('No loss is defined. Call FProp first.')
    return self._loss

  @property
  def train_op(self):
    assert self._train_op is not None, (
        'No train op is defined. Call BProp first.')
    return self._train_op

  @property
  def global_step(self):
    assert self._global_step is not None, ('No global_step is defined.')
    return self._global_step

  @property
  def input_generator(self):
    return self.input

  @property
  def eval_metrics(self):
    """Returns the evaluation metrics.

    Returns:
      A map from metric name (a python string) to a tuple (value, weight).
      Both value and weight are scalar Tensors.
    """
    return self._eval_metrics

  @property
  def per_example_tensors(self):
    """Returns per-example outputs.

    Returns:
      A map from tensor name (a python string) to a tensor, where the
      first dimension is the batch index of the training example corresponding
      to this output.
    """
    return self._per_example

  def AddEvalMetric(self, name, value, weight):
    """Adds a metric to the eval metrics.

    Args:
      name: A python string. The name of the metric.
      value: A scalar Tensor.
      weight: A scalar Tensor.

    Raises:
      ValueError: if `name` is already defined.

    """
    if name in self._eval_metrics:
      raise ValueError('Metric %s has already been defined.' % name)
    self._eval_metrics[name] = (value, weight)

  def AddPerExampleTensor(self, name, value):
    if name in self._per_example:
      raise ValueError('Metric %s has already been defined.' % name)
    self._per_example[name] = value

  @property
  def total_examples(self):
    """Returns the total number of training examples processed so far."""
    return self._total_examples.Value()

  @property
  def trainer_verbose_tensors(self):
    """Return the dict of verbose tensors to eval in the training loop."""
    return self._trainer_verbose_tensors

  def AddTrainerVerboseTensor(self, name, target):
    """Add a (set of) tensors to be evaluated in the training loop.

    Args:
      name: A python string. The name of the target(s).
      target: A Tensor or a list or dict of Tensors.

    Raises:
      ValueError: if `name` is already defined.

    """
    if name in self._trainer_verbose_tensors:
      raise ValueError('Verbose target %s has already been defined.' % name)
    self._trainer_verbose_tensors[name] = target

  def IncrementTotalSamples(self, value=None):
    """Updates the total number of training examples with the batch size."""
    p = self.params
    if self._total_examples is None:
      with tf.variable_scope(p.name):
        self._total_examples = StatsCounter('total_samples')
    if value is None:
      assert self.input_generator is not None, ('No input generator defined')
      value = self.input_generator.GlobalBatchSize()
    return self._total_examples.IncBy(p, value)

  def IncrementTotalNans(self, value):
    """Updates the total number of NaN/Inf gradients by `value`."""
    if self._total_nans_and_infs is None:
      with tf.variable_scope(
          py_utils.global_variable_scope, reuse=tf.AUTO_REUSE):
        self._total_nans_and_infs = StatsCounter('total_nan_gradients')
    return self._total_nans_and_infs.IncBy(self.params, value)

  def _UpdateVnConfig(self):
    """Update vn config from the various vn flags."""
    p = self.params
    tp = p.train
    if tp:
      vn_enabled = ((tp.vn_std > 0) and p.vn and
                    (p.vn.global_vn or p.vn.per_step_vn))
      if p.is_eval or (not vn_enabled):
        p.vn = py_utils.VariationalNoiseParams(None, False, False)
      else:
        # vn.scale is dependent on global_step.
        p.vn.scale = tf.cast(self.global_step > tp.vn_start_step,
                             py_utils.FPropDtype(p)) * tp.vn_std

  def _GetMaskUpdateOp(self):
    """Returns op to update masks and threshold variables for model pruning."""
    p = self.params
    tp = p.train
    mask_update_op = tf.no_op()
    if tp.pruning_hparams_dict:
      assert isinstance(tp.pruning_hparams_dict, dict)
      pruning_hparams = tf.contrib.model_pruning.get_pruning_hparams(
      ).override_from_dict(tp.pruning_hparams_dict)
      pruning_obj = tf.contrib.model_pruning.Pruning(
          pruning_hparams, global_step=self.global_step)
      pruning_obj.add_pruning_summaries()
      mask_update_op = pruning_obj.conditional_mask_update_op()
    return mask_update_op


class DistillationTask(BaseTask):
  """A task to distill knowledge from a teacher task to a student task.

  The training parameters (e.g., learning rate) are determined only by
  `DistillationTask.params.train`. Teacher and student task's training and eval
  parameters must be set to None.
  """

  @classmethod
  def Params(cls):
    p = super(DistillationTask, cls).Params()
    p.Define('teacher', None, 'The teacher task params.')
    p.Define('student', None, 'The student task params.')
    p.Define(
        'distillation_loss_weight',
        # Only uses distillation loss by default.
        lr_schedule.ConstantOne.Params(),
        'A schedule of distillation loss weight. '
        'The weight determines the fraction of total loss contributed by '
        'distillation loss, while the rest loss will be computed against '
        'the ground truth. '
        'A weight of 0 means to only use ground-truth and ignore teacher '
        'predictions, while a weight 1 means to only use teacher '
        'predictions and ignore ground truth. '
        'The weight is specified as a schedule to allow it to change '
        'during training.')
    p.Define(
        'teacher_target_type', 'truth', 'The target type for the teacher. '
        'Choices are: '
        ' "truth": using the ground-truth target labels '
        ' "beam": using the 1-best hypothesis from the beam search.')
    p.Define(
        'beam_search_temperature', 1.0, 'The temperature to scale the'
        'log-prob of each beam search hypothesis. This is used in '
        'training only')
    return p

  @base_layer.initializer
  def __init__(self, params):
    assert issubclass(params.cls, DistillationTask)
    super(DistillationTask, self).__init__(params)

    p = self.params
    # While student does not need its own input generator for training, it
    # needs an input generator for inference graphs.
    p.student.input = p.input
    # Teacher also might need an input generator, eg. for waveform_processor.
    p.teacher.input = p.input
    with tf.variable_scope(p.name):
      for child in ('teacher', 'student'):
        child_p = getattr(p, child)
        assert issubclass(child_p.cls, BaseTask)
        assert child_p.train is None
        assert child_p.eval is None
        # In theory it's ok for teacher to be a DistillationTask. In practice
        # it probably won't happen.
        assert not issubclass(child_p.cls, DistillationTask)
        child_p.name = child
        self.CreateChild(child, child_p)
      self.CreateChild('distillation_loss_weight', p.distillation_loss_weight)

  def ComputePredictions(self, theta, input_batch):
    p = self.params
    with tf.name_scope(p.name):
      if p.teacher_target_type == 'truth':
        teacher_predictions = self.teacher.ComputePredictions(
            theta.teacher, input_batch)
        student_predictions = self.student.ComputePredictions(
            theta.student, input_batch)
        return py_utils.NestedMap(
            teacher=teacher_predictions, student=student_predictions)
      elif p.teacher_target_type == 'beam':
        (teacher_predictions, teacher_input_batch,
         teacher_beam_prob) = self.teacher.ComputeBeamPredictions(
             theta.teacher, input_batch, p.beam_search_temperature)
        # We use 'teacher_input_batch' instead of 'input_batch' for 'student'
        # because the training of student network uses target transcripts for
        # the "teacher forcing" mode and here the target transcripts should come
        # from the teacher's beam search.
        student_predictions = self.student.ComputePredictions(
            theta.student, teacher_input_batch)
        return py_utils.NestedMap(
            teacher=teacher_predictions,
            student=student_predictions,
            teacher_beam_prob=teacher_beam_prob)
      else:
        raise ValueError('teacher target type not defined properly: %s' %
                         self.p.teacher_target_type)

  def ComputeLoss(self, theta, input_batch, predictions):
    per_example = {}
    with tf.name_scope('groundtruth_loss'):
      groundtruth_loss, groundtruth_per_example = self.student.ComputeLoss(
          theta.student, input_batch, predictions.student)
      groundtruth_loss['groundtruth_loss'] = groundtruth_loss['loss']
      per_example.update(groundtruth_per_example)

    with tf.name_scope('distillation_loss'):
      distillation_loss, distill_per_example = self.ComputeDistillationLoss(
          theta, input_batch, predictions)
      distillation_loss['distillation_loss'] = distillation_loss['loss']
      per_example.update(distill_per_example)

    distillation_loss_weight = self.distillation_loss_weight.FProp(
        theta.distillation_loss_weight, self.global_step)
    metrics = py_utils.CombineMetrics([
        (groundtruth_loss, 1 - distillation_loss_weight),
        (distillation_loss, distillation_loss_weight),
    ])
    return metrics, per_example

  def ComputeDistillationLoss(self, theta, input_batch, predictions):
    raise NotImplementedError('Abstract method')

  def BProp(self):
    # Only bprop on student variables.
    self._BPropForVariables(self.student.vars)

  def Decode(self, input_batch):
    return self.student.Decode(input_batch)

  def Inference(self):
    return self.student.Inference()

  def CreateDecoderMetrics(self):
    return self.student.CreateDecoderMetrics()

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    return self.student.PostProcessDecodeOut(dec_out_dict, dec_metrics_dict)


class BaseModel(base_layer.BaseLayer):
  """The abstract model class. All models are sub-class of this class."""

  @classmethod
  def Params(cls):
    p = super(BaseModel, cls).Params()
    p.Define(
        'model', None, 'Which python function generates the param. It includes '
        'the file name and lineno where the function is defined.')
    p.Define(
        'cluster', cluster_factory.Cluster.Params(),
        'The training cluster. Individual layer may config differently'
        ' based on training cluster it is running under.')
    p.Define('input', None, 'Input generator Params.')
    p.Define('build_data', build_data.BuildData(), 'Build data of this binary.')
    p.Define('train', hyperparams.Params(),
             'Params to control how this model should be trained.')
    tp = p.train
    tp.Define(
        'start_up_delay_steps', 200, 'i-th replica starts training after '
        'i*(i+1)/2*start_up_delay_steps steps')
    tp.Define('max_steps', 4 * 10**6, 'Training max of 4M steps.')
    tp.Define('tpu_steps_per_loop', 100, 'The number of training steps per '
              'training loop for TPUs.')
    tp.Define(
        'ema_decay', 0.0,
        'If > 0, enable ExponentialMovingAverage during training '
        'with the give decay. '
        'Must be < 1. Disabled if <= 0.')
    tp.Define('init_from_checkpoint_rules', {},
              'See BaseTask documentation for details.')
    tp.Define('early_stop', None,
              'Early stopping based on dev-set performance.')
    tp.Define('save_interval_seconds', 60 * 10,
              'Generates a checkpoint roughly once every this many seconds.')
    tp.Define('summary_interval_steps', 100,
              'Generates a checkpoint roughly once every this many steps.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes this Model."""
    assert issubclass(params.cls, BaseModel)
    self._global_step_var = py_utils.GetOrCreateGlobalStepVar()
    self._global_step = tf.identity(
        self._global_step_var, name='global_step_tensor')
    super(BaseModel, self).__init__(params)
    # tasks are not yet instantiated.
    self._total_examples_sum = None

    self._ema = None
    tp = self.params.train
    tf.logging.info('Training parameters for %s: %s', params.cls, tp)
    if tp.ema_decay > 0:
      assert tp.ema_decay < 1.0
      self._ema = tf.train.ExponentialMovingAverage(
          decay=tp.ema_decay, num_updates=self.global_step)

  @property
  def global_step(self):
    assert self._global_step is not None, ('No global_step is defined.')
    return self._global_step

  @property
  def ema(self):
    return self._ema

  def ConstructFPropBPropGraph(self):
    raise NotImplementedError('Abstract method')

  def ConstructFPropGraph(self):
    raise NotImplementedError('Abstract method')

  @property
  def tasks(self):
    """Returns a list of all tasks."""
    raise NotImplementedError('Abstract method')

  def GetTask(self, task_name):
    """Return the task associated with 'task_name'.

    Args:
      task_name: string, the name of the model task to be returned.

    Returns:
      An instance of `BaseTask`.
    """
    raise NotImplementedError('Abstract method')

  @property
  def total_examples(self):
    """Returns the total number of training examples processed so far."""
    if self._total_examples_sum is None:
      self._total_examples_sum = tf.reduce_sum(
          [task.total_examples for task in self.tasks])
    return self._total_examples_sum

  def ProcessFPropResults(self, sess, global_step, metrics, per_example):
    """Called once for each train loop.

    BaseTask.ProcessFPropResults is also called on each loop, so you
    can put your implementation wherever it is most convenient for you.

    Be sure to implement BaseTask.FilterPerExampleTensors if you plan to use any
    per-example tensors in this method.

    Args:
      sess: a session.
      global_step: approximate number of model training steps.
      metrics: the metrics dict returned by FPropTower.
      per_example: the per_example dict returned by FPropTower.
    """
    pass


class SingleTaskModel(BaseModel):
  """Model that consists of a single task."""

  @classmethod
  def Params(cls, task_params=None):
    p = super(SingleTaskModel, cls).Params()
    p.Define('task', None, 'Task Params.')

    if task_params is not None:
      # Copy over model parameters from the task parameters.
      p.task = task_params
      base_layer.BaseLayer.CopyBaseParams(p.task, p)
      tp = p.train
      tp.start_up_delay_steps = p.task.train.start_up_delay_steps
      tp.max_steps = p.task.train.max_steps
      tp.tpu_steps_per_loop = p.task.train.tpu_steps_per_loop
      tp.ema_decay = p.task.train.ema_decay
      # init_from_checkpoint_rules does not need to be copied.
      tp.early_stop = p.task.train.early_stop
      tp.save_interval_seconds = p.task.train.save_interval_seconds
      tp.summary_interval_steps = p.task.train.summary_interval_steps

    return p

  @base_layer.initializer
  def __init__(self, params):
    assert issubclass(params.cls, SingleTaskModel)
    assert params.task
    p = params.Copy()  # Make a copy to avoid modifying the input.
    p.name = p.name or p.task.name
    p.task.name = p.task.name or p.name
    if p.input:
      assert not p.task.input
      p.task.input = p.input
    else:
      assert p.task.input
      p.input = p.task.input

    super(SingleTaskModel, self).__init__(p)

    p = self.params
    with py_utils.GlobalStepContext(self.global_step):
      self.CreateChild('_task', p.task)

  @property
  def tasks(self):
    return [self._task]

  def GetTask(self, task_name=None):
    assert not task_name, 'Must not specify >task_name< for single-task model.'
    return self._task

  def SampleTask(self, global_step):
    return self._task

  def ConstructFPropBPropGraph(self):
    self._task.FPropDefaultTheta()
    self._task.BProp()
    if self.ema:
      tf.logging.info('ApplyExponentialMovingAverage on %s', self._task)
      self._task.ApplyExponentialMovingAverage(self.ema)

  def ConstructFPropGraph(self):
    self._task.FPropDefaultTheta()


class MultiTaskModel(BaseModel):
  """Model that consists of multiple tasks."""

  @classmethod
  def Params(cls):
    p = super(MultiTaskModel, cls).Params()
    p.Define('task_params', hyperparams.Params(),
             'Params object mapping task name to task Params.')
    p.Define(
        'task_probs', hyperparams.Params(),
        'Params object mapping task name to the relative likelihood the '
        'task will be sampled during training.')
    p.Define('task_schedule', None, 'Task schedule.')
    p.Define(
        'task_global_step', False,
        'Whether or not to use task-specific global steps, which causes each '
        'task to use its own global_step instead of the true global_step.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    assert issubclass(params.cls, MultiTaskModel)
    super(MultiTaskModel, self).__init__(params)
    p = self.params
    assert len(p.task_params) > 1

    # Pass input params to tasks.
    assert isinstance(p.input, hyperparams.Params)
    assert set(dir(p.input)) == set(dir(p.task_params))
    for k, v in p.task_params.IterParams():
      assert isinstance(v, hyperparams.Params)
      assert not v.input
      v.input = p.input.Get(k)

    # For compatibility with older API (with p.task_probs)
    if p.task_schedule is None:
      p.task_schedule = task_scheduler.ConstantScheduler.Params()
      p.task_schedule.task_probs = sorted(list(p.task_probs.IterParams()))

    # CreateChild copies over global configs in p to individual task params,
    # which then gets propagated down to all sub-layers during
    # BaseTask._PropagateDownGlobalConfigs(), or through sub-sequent CreateChild
    # or CreateChildren calls.
    with py_utils.GlobalStepContext(self.global_step):
      with tf.name_scope(p.name):
        sorted_task_params = sorted(
            (task_name, task_params)
            for task_name, task_params in p.task_params.IterParams())
        for task_name, task_params in sorted_task_params:
          if p.task_global_step:
            assert task_name == task_params.name
            CreateTaskGlobalStep(task_name)
          # Make sure each task is under its own variable scope.
          with tf.variable_scope(task_name):
            self.CreateChild(task_name, task_params)
        self.CreateChild('task_schedule', p.task_schedule)

  @property
  def task_names(self):
    sorted_task_names = sorted(
        task_name for task_name, _ in self.params.task_params.IterParams())
    return sorted_task_names

  @property
  def tasks(self):
    return [self.children[name] for name in self.task_names]

  def GetTask(self, task_name):
    assert task_name, 'Must specify >task_name< for multi-task model.'
    return self.children[task_name]

  def SampleTask(self, global_step):
    """Sample a task according self.task_schedule.

    `self.task_schedule.cur_probs` will also be updated.

    Args:
      global_step: int. Current time step.
    """
    sampled_task = self.task_schedule.Sample(global_step)
    tf.logging.info('Sampled task: %s', sampled_task)
    return self.children[sampled_task]

  def ConstructFPropBPropGraph(self):
    for task_name in self.task_names:
      with tf.name_scope(task_name):
        task = self.GetTask(task_name)
        task.FPropDefaultTheta()
        task.BProp()
        if self.ema:
          task.ApplyExponentialMovingAverage(self.ema)

  def ConstructFPropGraph(self):
    for task_name in self.task_names:
      with tf.name_scope(task_name):
        task = self.GetTask(task_name)
        task.FPropDefaultTheta()
