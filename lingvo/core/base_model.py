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


class BaseTask(base_layer.LayerBase):
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
        'task_global_step', False,
        'Whether or not to create a task-specific global step. '
        'When a task specific global step exists, learning rate schedule '
        'depends on the task specific global step, instead of the shared '
        'global step.')
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
    tp.Define('learning_rate', 0.0, 'learning rate to use.')
    tp.Define('clip_gradient_norm_to_value', 0.0,
              'Clip gradient norm to this value.')
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
        'initialize part of the model. Variables are only loaded form this '
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
    super(BaseTask, self).__init__(params)

    p = self.params

    if p.input:
      # TODO(zhifengc): Consider a simpler way to ensure the input
      # generator stops after one epoch.
      if p.is_eval and p.eval:
        seq_inp = issubclass(p.input.cls,
                             base_input_generator.BaseSequenceInputGenerator)
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

      cluster = cluster_factory.Current()
      with tf.device(cluster.input_device), py_utils.outside_all_rewrites():
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

    self._shared_global_step = py_utils.GetOrCreateGlobalStep()
    tp = p.train
    if tp:
      if tp.task_global_step:
        self._task_global_step = py_utils.CreateTaskGlobalStep(p, p.name)
        self._global_step = self._task_global_step
      else:
        self._task_global_step = None
        self._global_step = self._shared_global_step
      if tp.grad_norm_tracker:
        with tf.variable_scope(p.name):
          self.CreateChild('grad_norm_tracker', tp.grad_norm_tracker)

      self.CreateChild('lr_schedule', tp.lr_schedule)
    self._UpdateVnConfig()

  def ComputePredictions(self, theta, input_batch):
    """Computes predictions for 'input_batch'.

    The output can be in the form of probablistic distributions, e.g., softmax
    logits for discrete outputs, mixture of logistics for continuous values, or
    regression values.

    For training/evaluation, the output will be used for computing loss and
    gradient updates, including comparing predicted distributions between
    teacher and student for distillation. During inference the output can be
    used to compute final outputs, perhaps with sampling.

    Args:
      theta: A nested map object containing variable values of this task.
      input_batch: A nested map object containing input tensors to this tower.

    Returns:
      Predictions, in the form of a single Tensor, a NestedMap, or a namedtuple.
    """
    raise NotImplementedError('Abstract method')

  def ComputeLoss(self, theta, input_batch, predictions):
    """Computes loss and other metrics for the given predictions.

    Args:
      theta: A nested map object containing variable values of this task.
      input_batch: A nested map object containing input tensors to this tower.
      predictions: The output of ComputePredictions.

    Returns:
      A dict containing str keys and (metric, weight) pairs as values, where
      one of the keys is expected to be 'loss'.
    """
    raise NotImplementedError('Abstract method')

  def FPropTower(self, theta, input_batch):
    """Forward propagation through one tower of the model.

    Args:
      theta: A nested map object containing variable values of this
        task copied to this tower's devices.
      input_batch: A nested map object containing input tensors to this tower.

    Returns:
      A dict containing metrics pairs.
    """
    predicted = self.ComputePredictions(theta, input_batch)
    return self.ComputeLoss(theta, input_batch, predicted)

  def FProp(self, theta):
    """Forward propagation.

    This default FProp implementation here supports batch splitting in
    synchronous and asynchronous training when sub-classes implement
    FPropTower.

    Args:
      theta: A nested map object containing weights' values of this
        layer and its children layers.

    Returns:
      A dict containing metrics pairs. One of the keys should be 'loss' and its
      value should be a (loss, num_predictions) pair.
    """
    p = self.params
    # Casting theta_local leads to OOM on GPUs for large models, see comments
    # for WmtEnFrTransformerFloat16 and WmtEnFrTransformerBigFloat16.
    theta = py_utils.MaybeCastTheta(theta, p)

    cluster = cluster_factory.Current()

    with tf.name_scope('fprop'), tf.name_scope(p.name):
      all_fprop_metrics = []

      if py_utils.use_tpu():
        batch = self.input_generator.CreateTpuFeeds()
        with tf.name_scope('tower_0_0'):
          dec_metrics = self.FPropTower(theta, batch)
        all_fprop_metrics.append(dec_metrics)
      else:
        # Splits the input batch on the input device.
        num_splits = cluster.num_splits_per_client
        with tf.device(cluster.input_device):
          batches = self.input_generator.SplitInputBatch(num_splits)
          assert num_splits == len(batches)

        # dev_list_per_replica[i][j] is the i-th worker's j-th device.
        dev_list_per_replica = cluster.available_devices.tolist()

        # Asserts invariant of the total number of splits w.r.t.,
        # splits per worker.
        splits_per_replica = cluster.num_splits_per_replica
        assert num_splits == splits_per_replica * len(dev_list_per_replica)

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
                      batches[split_id])
                  dec_metrics = self.FPropTower(theta_local, batch)
            all_fprop_metrics.append(dec_metrics)

      metrics = py_utils.WeightedAvgOfMetrics(all_fprop_metrics)

    # Adds stats about the input batch.
    metrics['num_samples_in_batch'] = (self.input_generator.InputBatchSize(),
                                       tf.constant(1.0))
    # Generates summaries.
    for name, (value, weight) in six.iteritems(metrics):
      self.AddEvalMetric(name, value, weight)

    # Loss.
    self._loss, self._num_predicts = metrics['loss']
    self._loss = py_utils.CheckNumerics(self._loss)

    return metrics

  def FPropDefaultTheta(self):
    """Calls FProp."""
    return self.FProp(self.theta)

  def AdjustGradients(self, vars_gradients):
    """Allow for custom gradient manipulation prior to clipping."""
    return vars_gradients

  def BProp(self):
    """Constructs the backward graph."""
    p = self.params
    vs = self.vars
    if p.train.bprop_variable_filter:

      def VariableFilter(v):
        if re.search(p.train.bprop_variable_filter, v.name):
          return True
        tf.logging.info('bprop disabled by bprop_variable_filter: %s', v.name)
        return False

      vs = vs.Filter(VariableFilter)
      tf.logging.info('Filtered bprop variables: %s', vs)
    self._BPropForVariables(vs)

  def _HasNanOrInf(self, var_grads):
    """Returns a bool tensor to indicate if var_grads contains NaNs or Infs.

    Args:
      var_grads: A NestedMap with (var, grad) tuple as the map value.

    Returns:
      A bool scalar tensor to indicate if the var_grads contains NaNs or Infs.
    """

    def HasNanOrInf(x):
      with tf.device(x.device):
        return tf.reduce_any(tf.logical_or(tf.is_nan(x), tf.is_inf(x)))

    return tf.reduce_any([(HasNanOrInf(g.values) if isinstance(
        g, tf.IndexedSlices) else HasNanOrInf(g))
                          for (_, g) in var_grads.Flatten()])

  def ScaleGradients(self, var_grads):
    """Scales gradients according to training params.

    Args:
      var_grads: a NestedMap whose values are (var, grad) pairs.

    Returns:
      (has_nan_or_inf, grad_scale, final_var_grads), where:
        has_nan_or_inf: a scalar of 0 or 1, indicating whether there is any NaN
          or Inf in input gradients.
        grad_scale: the gradient scale. 0 if gradient updates should be skipped
          for the step.
        final_var_grads: a NestedMap whose values are (var, grad) pairs, where
          gradients have already been scaled.
    """
    p = self.params
    tp = p.train

    # Computes gradients' norm and adds their summaries. Note that all_grad_norm
    # may be nan, which may cause grad_scale to be nan.
    _, all_grad_norm = summary_utils.AddNormSummary(p, 'all', var_grads)
    grad_norm_is_nan_or_inf = tf.logical_or(
        tf.is_nan(all_grad_norm), tf.is_inf(all_grad_norm))

    # Optional gradient adjustment. Note that this happens after computing
    # all_grad_norm.
    var_grads = self.AdjustGradients(var_grads)

    # Handles NaN/Inf gradients.
    has_nan_or_inf = self._HasNanOrInf(var_grads)
    # Grad norm can still be inf even if none of the individual grad is inf.
    has_nan_or_inf = tf.logical_or(has_nan_or_inf, grad_norm_is_nan_or_inf)

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

    summary_utils.scalar(p, 'grad_scale_all', grad_scale)
    final_var_grads = py_utils.ApplyGradMultiplier(var_grads, grad_scale)
    return has_nan_or_inf, grad_scale, final_var_grads

  def _BPropForVariables(self, vmap):
    """Constructs the backward graph for the given variables.

    Args:
      vmap: a NestedMap of variables.
    """
    p = self.params
    tp = p.train

    # Compute gradients.
    self._var_grads = py_utils.ComputeGradients(self.loss, vmap)

    # L2 regularizer.
    if tp.l2_regularizer_weight is not None:
      l2_loss, self._var_grads = py_utils.AdjustGradientsWithL2Loss(
          self._var_grads, tp.l2_regularizer_weight)
      summary_utils.scalar(p, 'l2_loss', l2_loss)

    # Histogram summary.
    summary_utils.CollectVarHistogram(p, self._var_grads)

    # Apply gradient clipping.
    has_nan_or_inf, _, self._var_grads = self.ScaleGradients(self._var_grads)

    lrs = self.lr_schedule.Value(self._global_step)
    summary_utils.scalar(p, 'lr_schedule', lrs)
    lr = tp.learning_rate * lrs

    opt = tp.optimizer.cls(tp.optimizer.Copy().Set(add_summary=p.add_summary))
    var_update_op = opt.Apply(lr, self._var_grads)

    increment_global_step_ops = []
    with tf.colocate_with(self._shared_global_step):
      increment_global_step_ops.append(
          tf.assign_add(self._shared_global_step, 1))
    if self._task_global_step:
      with tf.colocate_with(self._task_global_step):
        increment_global_step_ops.append(
            tf.assign_add(self._task_global_step, 1))
    increment_global_steps = tf.group(*increment_global_step_ops)

    relevant_bn_updates, _ = py_utils.FindRelevantBatchNormUpdates(
        self.loss, tf.get_collection(py_utils.BATCH_NORM_UPDATES))
    batch_norm_updates = tf.group(*relevant_bn_updates)

    # Update stats.
    stats_updates = tf.group(
        self.IncrementTotalSamples(),
        self.IncrementTotalNans(tf.to_int32(has_nan_or_inf)))

    # Post training step update.
    post_training_step_updates = self.PostTrainingStepUpdate(self._global_step)

    # Get the op to update the weight masks and thresholds
    mask_update_op = self._GetMaskUpdateOp()

    # TODO(rpang): try to structure _train_op as:
    #   tf.cond(skip_step, <only update skip stats>, <all updates>)
    # so that we skip all other updates when a step is skipped.
    self._train_op = tf.group(
        var_update_op,
        batch_norm_updates,
        stats_updates,
        post_training_step_updates,
        increment_global_steps,
        mask_update_op,
        name='train')

  def ApplyExponentialMovingAverage(self, ema):
    """Wraps self.train_op with an Op to update exponential moving average."""
    # We need to apply EMA to all trainable variables of this Task, not just
    # bprop vars, so that we create a shadow '/ExponentialMovingAverage'
    # variable for every trainable variable.
    all_vars = set(self.vars.Flatten()) & set(tf.trainable_variables())
    for var in all_vars:
      tf.logging.debug('ApplyExponentialMovingAverage: %s', var.name)
    with tf.control_dependencies(
        [self._train_op]), tf.name_scope('moving_average'):
      self._train_op = ema.apply(all_vars)

  def Decode(self):
    """Constructs the inference graph for eval decoding.

    Returns a dict of tensors as decoder output.
    """
    pass

  def Inference(self):
    """Constructs the inference graph.

    Each subgraph represents a public API for a part of the graph which can
    be operated independently. By convention, the subgraph named 'default'
    should perfom end to end inference via the input generator.

    Note that having distinct subgraphs (e.g. 'encoder', 'decoder') is
    not just a space optimization: when driving the graph externally in an
    online fashion, evaluation often needs to be broken into pieces. In this
    case, the graph will be constructed with only those pieces.

    Returns:
      An inference_graph_pb2.InferenceGraph message or a legacy
      dict of {'subgraph_name': (fetches, feeds)} for each subgraph where
      feeds and fetches are NestedMaps of public API names to internal
      Tensor instances. Callers are responsible for handling either format.
    """
    raise NotImplementedError('Abstract method')

  def CreateDecoderMetrics(self):
    """Creates a dict of decoder metrics to be updated via PostProcessDecodeOut.

    Returns a dict mapping from string keys to BaseMetric objects.
    """
    pass

  def PostProcessDecodeOut(self, decode_out_dict, decode_metrics_dict):
    """Post-processes decoder out, and updates contents of decode_metrics_dict.

    Args:
      decode_out_dict: A dictionary of Tensors fetched.
      decode_metrics_dict: A dict mapping from string key to BaseMetric object
        as created by CreateDecoderMetrics.

    Returns:
      output_key_value_pairs: a list of (key, value) pairs that can be saved
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

  def AddEvalMetric(self, name, value, weight):
    """Adds a metric to the eval metrics.

    Args:
      name: A python string. The name of the metric.
      value: A scalar Tensor.
      weight: A scalar Tensor.

    Raises:
      ValueError: if 'name' is already defined.

    """
    if name in self._eval_metrics:
      raise ValueError('Metric %s has already been defined.' % name)
    summary_utils.scalar(self.params, name, value)
    self._eval_metrics[name] = (value, weight)

  @property
  def total_examples(self):
    """Returns the total number of training examples processed so far."""
    return self._total_examples.Value()

  def IncrementTotalSamples(self, value=None):
    """Updates the total number of training examples with the batch size."""
    p = self.params
    if self._total_examples is None:
      with tf.variable_scope(p.name):
        self._total_examples = py_utils.StatsCounter('total_samples')
    if value is None:
      assert self.input_generator is not None, ('No input generator defined')
      value = self.input_generator.InputBatchSize()
    return self._total_examples.IncBy(p, value)

  def IncrementTotalNans(self, value):
    """Updates the total number of NaN/Inf gradients by 'value'."""
    if self._total_nans_and_infs is None:
      with tf.variable_scope(
          py_utils.global_variable_scope, reuse=tf.AUTO_REUSE):
        self._total_nans_and_infs = py_utils.StatsCounter('total_nan_gradients')
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
        p.vn.scale = tf.cast(self._global_step > tp.vn_start_step,
                             p.dtype) * tp.vn_std

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
          pruning_hparams, global_step=self._global_step)
      pruning_obj.add_pruning_summaries()
      mask_update_op = pruning_obj.conditional_mask_update_op()
    return mask_update_op


class DistillationTask(BaseTask):
  """A task to distill knowledge from a teacher task to a student task.

  The training parameters (e.g., learning rate) are determined only by
  DistillationTask.params.train. Teacher and student task's training and eval
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
        lr_schedule.PiecewiseConstantLearningRateSchedule.Params().Set(
            boundaries=[], values=[1.0]),
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
    with tf.name_scope('groundtruth_loss'):
      groundtruth_loss = self.student.ComputeLoss(theta.student, input_batch,
                                                  predictions.student)
      groundtruth_loss['groundtruth_loss'] = groundtruth_loss['loss']

    with tf.name_scope('distillation_loss'):
      distillation_loss = self.ComputeDistillationLoss(theta, input_batch,
                                                       predictions)
      distillation_loss['distillation_loss'] = distillation_loss['loss']

    distillation_loss_weight = self.distillation_loss_weight.FProp(
        theta.distillation_loss_weight, self._global_step)
    metrics = py_utils.CombineMetrics([
        (groundtruth_loss, 1 - distillation_loss_weight),
        (distillation_loss, distillation_loss_weight),
    ])
    return metrics

  def ComputeDistillationLoss(self, theta, input_batch, predictions):
    raise NotImplementedError('Abstract method')

  def BProp(self):
    # Only bprop on student variables.
    self._BPropForVariables(self.student.vars)

  def Decode(self):
    return self.student.Decode()

  def Inference(self):
    return self.student.Inference()

  def CreateDecoderMetrics(self):
    return self.student.CreateDecoderMetrics()

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    return self.student.PostProcessDecodeOut(dec_out_dict, dec_metrics_dict)


class BaseModel(base_layer.LayerBase):
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
    super(BaseModel, self).__init__(params)
    self._global_step = py_utils.GetOrCreateGlobalStep()
    # tasks are not yet instantiated.
    self._total_examples_sum = None

    self._ema = None
    tp = self.params.train
    tf.logging.info('Training parameters for %s: %s', params.cls, tp)
    if tp.ema_decay > 0:
      assert tp.ema_decay < 1.0
      self._ema = tf.train.ExponentialMovingAverage(
          decay=tp.ema_decay, num_updates=self._global_step)

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
      An instance of BaseTask.
    """
    raise NotImplementedError('Abstract method')

  @property
  def total_examples(self):
    """Returns the total number of training examples processed so far."""
    if self._total_examples_sum is None:
      self._total_examples_sum = tf.reduce_sum(
          [task.total_examples for task in self.tasks])
    return self._total_examples_sum


class SingleTaskModel(BaseModel):
  """Model that consists of a single task."""

  @classmethod
  def Params(cls):
    p = super(SingleTaskModel, cls).Params()
    p.Define('task', None, 'Task Params.')
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
    with tf.name_scope(p.name):
      sorted_task_params = sorted(
          (task_name, task_params)
          for task_name, task_params in p.task_params.IterParams())
      for task_name, task_params in sorted_task_params:
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
