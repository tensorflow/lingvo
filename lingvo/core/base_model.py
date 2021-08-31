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
"""Base model."""

import collections
import re

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import base_layer
from lingvo.core import build_data
from lingvo.core import cluster_factory
from lingvo.core import early_stop
from lingvo.core import hyperparams
from lingvo.core import learner
from lingvo.core import optimizer
from lingvo.core import pruning_utils
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import summary_utils
from lingvo.core import task_scheduler
from lingvo.core import tpu_embedding_layers
from lingvo.core import decoder_lib
from lingvo.core import input_policy
from model_pruning.python import pruning


class DecodeFinalizeArgs(
    collections.namedtuple('DecodeFinalizeArgs',
                           ['decode_out_path', 'decode_out'])):
  """Arguments to BaseTask.DecodeFinalize().

  Attributes:
   decode_out_path: Path to where decoder outputs can be written.
   decode_out: A list of key value pairs aggregated from return values of.
     PostProcessDecodeOut().
  """


def _VariablesForEMA(params, model_var_list):
  """Gets a list of variables that need to apply exponential moving average."""
  # Use variable reference since variable is not hashable in eager mode.
  ref_set = lambda variables: set([v.ref() for v in variables])

  trainable_variables = [var for var in model_var_list if var.trainable]

  # We need to apply EMA to trainable and moving average variable of the task,
  # not just bprop vars, so that we create a shadow '/ExponentialMovingAverage'
  # variable for every trainable and moving average variable.
  all_refs = ref_set(trainable_variables) | ref_set(
      tf.moving_average_variables())
  if params.train.ema_decay_moving_vars:
    all_refs |= ref_set(tf.get_collection('moving_vars'))
  all_refs &= ref_set(model_var_list)

  # Remove TPU embedding variables since TPU embedding doesn't support EMA.
  tpu_embedding_vars = (
      tpu_embedding_layers.TpuEmbeddingCollection.Get().table_variables)
  if tpu_embedding_vars.Flatten():
    tf.logging.warning(
        'Detected TPU embedding variables, and EMA does not apply to them. '
        f'List of TPU embedding variables: {tpu_embedding_vars}.')
    all_refs -= ref_set(tpu_embedding_vars.Flatten())

  all_vars = [v.deref() for v in all_refs]
  for var in all_vars:
    tf.logging.debug('Variables for EMA: %s', var.name)
  return all_vars


class BaseTask(base_layer.BaseLayer):
  """A single encoder/decoder task.

  One task usually consists of one InputGenerator, one train_op,
  a list of eval_metrics, etc.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input', None, 'Input generator Params.')
    p.Define('encoder', None, 'Encoder Params.')
    p.Define('online_encoder', None, 'Online Encoder Params.')
    p.Define('decoder', None, 'Decoder Params.')
    p.Define(
        'task_global_step', False,
        'Whether or not to use task-specific global steps, which causes each '
        'task to use its own global_step instead of the true global_step. '
        'NOTE: this may be severely broken. Verify carefully!')
    p.Define(
        'defer_global_step_update', False,
        'Whether or not to defer the global step update. This is used when '
        'doing gradient accumulation, which update the global step only when '
        'weights are updated. Currently this supports only true global step.')
    p.Define('train', hyperparams.Params(),
             'Params to control how this task should be trained.')

    p.Define('ml_perf', hyperparams.Params(), 'MlPerf configuration.')

    tp = p.train
    tp.Define(
        'start_up_delay_steps', 200, 'i-th replica starts training after '
        'i*(i+1)/2*start_up_delay_steps steps')
    tp.Define('max_steps', 4 * 10**6, 'Maximum number of training steps.')
    tp.Define(
        'tpu_steps_per_loop', 1000, 'The number of training steps per '
        'training loop for TPUs. Note that this is not used by '
        'ExecutorTpu, which relies on ProgramSchedule.')
    tp.Define(
        'tpu_device_order_mode', None,
        'A device_assignment_lib.DeviceOrderMode enum that determines whether '
        'to assign devices in a way that the order of replicas or '
        'model-parallel cores will form a ring or mesh, or let the library to '
        'choose. Default None to AUTO.')
    tp.Define(
        'tpu_computation_shape', None,
        'A 4-element list that describes how virtual cores (which we specify '
        'in TF computation) should be mapped to one or more logical cores.')
    tp.Define(
        'vn_start_step', 200000000,
        'Step starting from which variational noise is added to '
        'params values during training.')
    tp.Define('vn_std', 0.0, 'Std of the variational noise.')
    tp.Define('early_stop', early_stop.EarlyStop.Params(),
              'Early stopping based on dev-set performance.')
    tp.Define(
        'ema_decay', 0.0,
        'If > 0, enable ExponentialMovingAverage during training '
        'with the give decay. '
        'Must be < 1. Disabled if <= 0. '
        'Note that TPU embedding does not support EMA, so if used together, '
        'there will be a mix of EMA and non-EMA variables in the model and the '
        'quality may be affected, so use them together at your own risk.')
    tp.Define(
        'ema_decay_moving_vars', None,
        'If True, include variables from collection "moving_vars" in ema.')
    tp.Define(
        'init_from_checkpoint_rules', {},
        'If not None, a dictionary with keys corresponding to a checkpoint '
        'path and values corresponding to variable loading rules is expected. '
        'Each key is expected to be a path to a checkpoint from which to '
        'initialize part of the model. Variables are only loaded from this '
        'path during initialization and will override values provided by '
        'initialization.\n'
        'The corresponding values (loading_rules) are expected to be a tuple '
        'consisting of two list: loading rules, and ignore rules, respectively.'
        'The first list (loading rules) contains the list of variables '
        'which should be initialized from the checkpoint: each element in the '
        'list is a pair of strings. The first element is a regex and the '
        'second is a python format string. If a variable in the model matches '
        'a regex, we rename using the format string to determine the '
        'corresponding var in the checkpoint. If a model variable would match '
        'multiple loading rules, the first rule that matches is used.\n'
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
        'init_from_checkpoint_override', '',
        'If set, override keys in init_from_checkpoint_rules with this. '
        'Once set, only one key is expected in '
        'init_from_checkpoint_rules. This is for easier param override '
        'when using --model_params_override or in xm.')
    tp.Define(
        'pruning_hparams_dict', None, 'Pruning related hyperparameters. A dict '
        'with hyperparameter: value pairs. See google-research.model_pruning.')
    tp.Define(
        'enqueue_max_steps', -1, 'Max enqueue steps. -1 meaning no limit.'
        ' This flag should be set for unit-test only.')
    tp.Define('save_interval_seconds', 60 * 10,
              'Generates a checkpoint roughly once every this many seconds.')
    tp.Define(
        'save_interval_steps', None,
        'Generates a checkpoint roughly once every this many training '
        'steps. Supersedes save_interval_seconds if not None.')
    tp.Define('save_max_to_keep', 100,
              'Maximum number of recent checkpoints to keep.')
    tp.Define('save_keep_checkpoint_every_n_hours', 0.5,
              'How often to keep a checkpoint.')
    tp.Define('async_checkpointing', False,
              'Checkpointing asynchronously. Currently only support executor.')
    tp.Define(
        'keep_per_example_loss', False,
        'If True, checks if per-example metrics contain a key named \'loss\', '
        'and if so copies it to the main metrics dictionary under key '
        '\'per_example_loss\'.')
    tp.Define('summary_interval_steps', 100,
              'Generates a summary roughly once every this many steps.')
    # The following params must mirror those in Learner.Params().
    # TODO(rpang): migrate existing params to use learner and
    # delete legacy params.
    # LINT.IfChange
    tp.Define(
        'learner', None, 'One or a list of optimization programs. '
        'If None, uses a Learner created from the legacy params '
        'defined below: learning_rate, lr_schedule, optimizer, etc.')
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
        'the behaviour of tf.clip_by_global_norm, if you are looking for '
        'tf.clip_by_norm refer to clip_gradient_single_norm_to_value. Note '
        'these are mutually exclusive.')
    tp.Define(
        'clip_gradient_single_norm_to_value', 0.0,
        'Clip gradient by single tensor norm to this value. This is '
        'similar to the behaviour of tf.clip_by_norm. Note this is mutually '
        'exlusive to using clip_gradient_norm_to_value.')
    tp.Define('grad_norm_to_clip_to_zero', 0.0,
              'Clip gradient to 0 if its norm exceeds this value.')
    tp.Define('grad_norm_tracker', None, 'Params for GradNormTracker.')
    tp.Define('optimizer', optimizer.Adam.Params(), 'Params for the optimizer.')
    tp.Define('lr_schedule', schedule.ContinuousSchedule.Params(),
              'Learning rate decay schedule.')
    tp.Define(
        'bprop_variable_filter', None,
        'If set, only backprop variables whose names partially match '
        'this regexp (re.search).')
    tp.Define(
        'bprop_variable_exclusion', None,
        'If set, do not backprop variables whose names partially match '
        'this regexp (re.search).')
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
    tp.Define('scale_gradients', True,
              'Whether to apply gradients adjustment and scaling.')
    tp.Define(
        'learner_use_variable_scope', True,
        'Create children of learner in tf.variable_scope. This may need '
        'to be set to False for compatibility with the existing '
        'checkpoints trained from legacy code. New models should always '
        'set this to True.')
    # LINT.ThenChange(learner.py)
    p.Define('eval', hyperparams.Params(),
             'Params to control how this task should be evaled.')
    ep = p.eval
    ep.Define(
        'samples_per_summary', 1000,
        'If > 0, generates one summary after this many samples, at most. '
        'If == 0 or the dataset has fewer examples, evaluate the whole set.')
    ep.Define(
        'decoder_samples_per_summary', None,
        'If > 0, each decoder summary will contain at most this many samples. '
        'If None, defaults to the actual value of `p.eval.samples_per_summary` '
        'for backwards compatibility.')
    ep.Define(
        'load_checkpoint_from', '',
        'If not Empty, specifies a location for the checkpoint that '
        'should be used for eval. One example format is a '
        'checkpoint directory of a training run.')
    ep.Define('start_eval_after', 0,
              'Start evaluation after specified number of steps.')
    ep.Define('start_decoder_after', 0,
              'Only decode checkpoints after this step.')
    ep.Define(
        'eval_all_checkpoints', False,
        'Compute evaluation metrics for every checkpoint saved by the Trainer.')
    ep.Define(
        'decode_all_checkpoints', False,
        'Compute decoder metrics for every checkpoint saved by the Trainer.')
    return p

  @classmethod
  def UpdateTargetVocabSize(cls, p, vocab_size, wpm_model=None):
    """Updates params with the vocab size and wpm model.

    Args:
      p: model params.
      vocab_size: size of the vocabulary.
      wpm_model: file name prefix pointing to a wordpiece model.

    Returns:
      Model params updated with the vocab size and wpm model.
    """
    dp = p.decoder
    p.decoder = dp.cls.UpdateTargetVocabSize(dp, vocab_size, wpm_model)
    return p

  def __init__(self, params):
    tp = params.train
    if tp and tp.init_from_checkpoint_override:
      assert len(tp.init_from_checkpoint_rules) == 1
      rules = list(tp.init_from_checkpoint_rules.values())[0]
      tp.init_from_checkpoint_rules.clear()
      tp.init_from_checkpoint_rules[tp.init_from_checkpoint_override] = rules
    assert issubclass(params.cls, BaseTask)
    # Ensure global_step exists before calling super.
    py_utils.GetOrCreateGlobalStepVar()
    super().__init__(params)

    p = self.params

    self._encoder = None
    self._online_encoder = None
    self._decoder = None

    self._loss = None
    self._train_op = None
    self._post_train_ops = []
    self._eval_metrics = {}
    self._per_example = {}

    # Create the gradient mask,
    self._per_input_gradient_mask = None

    if p.task_global_step:
      with tf.name_scope(None), tf.variable_scope(
          py_utils.GetGlobalVariableScope()):
        var_name = p.name + '_global_step'
        # Create the variable immediately.
        self._CreateVariableInternal(
            var_name,
            base_layer.CreateVariableMeta(
                var_params=py_utils.WeightParams(
                    [], py_utils.WeightInit.Constant(0), tf.int64),
                kwargs=dict(
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES])))
        summary_utils.scalar(var_name, self._private_vars[var_name])
        self._global_step_var = self._private_vars[var_name]
    else:
      self._global_step_var = py_utils.GetOrCreateGlobalStepVar()

    if p.input:
      # TODO(zhifengc): Consider a simpler way to ensure the input
      # generator stops after one epoch.
      if self.do_eval and p.eval:
        seq_inp = issubclass(p.input.cls,
                             base_input_generator.BaseInputGeneratorFromFiles)
        if p.input.num_samples > 0:
          if (p.eval.samples_per_summary == 0) or (p.input.num_samples <
                                                   p.eval.samples_per_summary):
            p.eval.samples_per_summary = p.input.num_samples
            # If we know the dataset size and we want to evaluate the full
            # set, we need to coordinate the input generator to flush out
            # all samples so the evaler and decoder compute metrics on the
            # whole set for each summary step.
            if seq_inp:
              p.input.flush_every_n = p.input.num_samples
          if p.eval.decoder_samples_per_summary is not None and (
              p.eval.decoder_samples_per_summary > p.input.num_samples):
            p.eval.decoder_samples_per_summary = p.input.num_samples
        if p.input.eval_samples_per_summary is not None:
          p.eval.samples_per_summary = p.input.eval_samples_per_summary
        if p.input.decoder_samples_per_summary is not None:
          p.eval.decoder_samples_per_summary = (
              p.input.decoder_samples_per_summary)
        if p.input.num_samples == 0 and not p.input.resettable:
          # Dataset size is unknown. Computes eval summary based on num_samples.
          # We require static dataset size for non-resettable inputs.
          assert p.eval.samples_per_summary > 0
        if seq_inp and p.input.num_batcher_threads > 1:
          tf.logging.warning('input.num_batcher_threads > 1 inside eval mode.  '
                             'The input generator may not iterate over exactly '
                             'one epoch per run')
      input_params = input_policy.Apply(p.input)
      tf.logging.info('input_params: %s', input_params)
      self.CreateChild('input', input_params)

    tp = p.train

    # p.train can be None if this task is the teacher/student task in a
    # DistillationTask.
    if tp:
      self._SetLearnerFromLegacyParams(tp)
      if tp.learner is not None:
        if isinstance(tp.learner, (list, tuple)):
          self.CreateChildren('learners', tp.learner)
        else:
          self.CreateChildren('learners', [tp.learner])
    self._UpdateVnConfig()

    if (tp and tp.pruning_hparams_dict and
        pruning_utils.UsePruningInterface(tp.pruning_hparams_dict)):
      pruning_utils.PruningOp.Setup(tp.pruning_hparams_dict, self.global_step)

  def InstantiateVariables(self):
    with py_utils.GlobalStepContext(self._global_step_var):
      super().InstantiateVariables()

  def _SetLearnerFromLegacyParams(self, tp):
    """Sets tp.learner based on legacy params."""
    if tp.learner is not None:
      return
    tp.learner = learner.ExtractLearnerFromLegacyParams(tp)

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

  def ComputeLoss(self, theta, predictions, input_batch):
    """Computes loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      (dict, dict):

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
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

    Be sure to implement BaseTask.FilterPerExampleTensors if you plan to use any
    per-example tensors in this method.

    Args:
      sess: a session.
      global_step: task global step. Since ProcessFPropResults is called after
        sess.run(train_op), this value will be p.train.tpu_steps_per_loop higher
        than the value in FProp.
      metrics: the metrics dict returned by FPropTower.
      per_example: the per_example dict returned by FPropTower.
    """
    p = self.params
    tp = p.train
    if not tp.pruning_hparams_dict:
      return
    if pruning_utils.PruningOp.ApplyPythonUpdate():
      pruning_utils.PruningOp.RunPythonUpdate(sess, global_step)

  def FPropTower(self, theta, input_batch):
    """Forward propagation through one tower of the model.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      (dict, dict):

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    predictions = self.ComputePredictions(theta, input_batch)
    return self.ComputeLoss(theta, predictions, input_batch)

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
      (dict, dict):

      - A dict containing str keys and (metric, weight) pairs as values, where
        one of the keys is expected to be 'loss'.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(
        p.name), py_utils.TaskCallScope(self):
      with py_utils.GlobalStepContext(self._global_step_var):
        # Always reset step seed at the start of a new global_step.
        py_utils.ResetStepSeed()
        metrics, per_example = self._FPropSplitInputBatch(theta, input_batch)
        self._FPropResult(metrics, per_example)
    return metrics, per_example

  def _FPropTpu(self, theta, input_batch):
    with tf.name_scope('tower_0_0'):
      metrics, per_example = self.FPropTower(theta, input_batch)
      metrics = py_utils.WeightedAvgOfMetrics([metrics])
    return metrics, per_example

  def _FPropSplitInputBatch(self, theta, input_batch):
    """Splits the input batch on the input device."""
    if py_utils.use_tpu():
      return self._FPropTpu(theta, input_batch)

    num_splits = self.cluster.num_splits_per_client

    if not isinstance(input_batch, list):
      input_batch = [input_batch]

    assert len(input_batch) == num_splits, (len(input_batch), num_splits)

    # dev_list_per_replica[i][j] is the i-th worker's j-th device.
    dev_list_per_replica = self.cluster.available_devices.tolist()

    # Asserts invariant of the total number of splits w.r.t.,
    # splits per worker.
    splits_per_replica = self.cluster.num_splits_per_replica
    assert num_splits == splits_per_replica * len(dev_list_per_replica), (
        num_splits, splits_per_replica, len(dev_list_per_replica))

    all_metrics = []
    all_per_example_tensors = []
    with self.cluster:
      for w_id, w_devs in enumerate(dev_list_per_replica):
        # Make local copy of the vars, shard on devices for this worker.
        theta_local = py_utils.CreateLocalTheta(
            theta, w_devs, label='worker %d' % w_id)

        for s_id in range(splits_per_replica):
          # s_id-th split for the w_id-th worker.
          split_id = splits_per_replica * w_id + s_id
          with cluster_factory.SetModelSplit(split_id) as c:
            with tf.device(c.WorkerDeviceInModelSplit(0)):
              with tf.name_scope('tower_%d_%d' % (w_id, s_id)):
                batch = input_batch[split_id]
                metrics, per_example = self.FPropTower(theta_local, batch)
          all_metrics.append(metrics)
          all_per_example_tensors.append(per_example)
    return py_utils.WeightedAvgOfMetrics(
        all_metrics), py_utils.ConcatPerExampleTensors(all_per_example_tensors)

  def _FPropResult(self, metrics, per_example):
    # Adds stats about the input batch.
    p = self._params
    if p.input is not None and 'num_samples_in_batch' not in metrics:
      metrics['num_samples_in_batch'] = (tf.convert_to_tensor(
          self.input_generator.GlobalBatchSize()), tf.constant(1.0))
    # Generates summaries.
    for name, (value, weight) in metrics.items():
      self.AddEvalMetric(
          name,
          value,
          weight,
          raise_if_already_added=not py_utils.IsEagerMode())
    if p.train.keep_per_example_loss and 'loss' in per_example:
      metrics['per_example_loss'] = per_example['loss']
    per_example = self.FilterPerExampleTensors(per_example)
    for name, value in per_example.items():
      self.AddPerExampleTensor(name, value)
    # Loss.
    self._loss, num_predictions = metrics['loss']
    self._loss = py_utils.CheckNumerics(self._loss)
    self._metrics = metrics
    if 'num_predictions' not in metrics:
      summary_utils.scalar('num_predictions', num_predictions)

  def GetInputBatch(self):
    """Gets an input batch."""
    if py_utils.use_tpu():
      return self.input_generator.TpuDequeueBatch()
    else:
      return self.input_generator.SplitInputBatch(
          self.cluster.num_splits_per_client)

  def FPropDefaultTheta(self, input_batch=None):
    """Calls `FProp` with this layer's parameters."""
    if input_batch is None:
      input_batch = self.GetInputBatch()
    return self.FProp(self.theta, input_batch)

  def AdjustGradients(self, vars_gradients):
    """Allow for custom gradient manipulation prior to clipping."""
    tf.logging.info('BaseTask.AdjustGradients')
    return vars_gradients

  def PostTrainingLoop(self, outfeed=None):
    """Construct the post training loop op.

    Args:
      outfeed: a dict of tensors dequeued from TPU outfeed queue.
    """
    with py_utils.GlobalStepContext(self._global_step_var):
      self._post_training_loop_op = tf.group(
          *[opt.ApplyPostTrainingLoop() for opt in self.learners])

  def BProp(self):
    with py_utils.GlobalStepContext(
        self._global_step_var), py_utils.TaskCallScope(self):
      self._BPropForVariables(self.vars)

  def _BPropGenTrainOps(self, vmap, metrics=None, add_summary=True):
    """Populates the train_ops dictionary in a backwards pass."""
    metrics = metrics or self._metrics

    bprop_variable_filters = self.input_generator.GetBpropVariableFilters()
    # Only compute the mask if the variable filters are not empty.
    if bprop_variable_filters != [''] * len(bprop_variable_filters):
      self._ComputeGradientMask(bprop_variable_filters)
    train_ops = {}  # mapping from op name to op.
    gradient_mask = None
    if self._per_input_gradient_mask:
      # TODO(neerajgaur): Change this to use source_selected from input_batch.
      onehot = self.input_generator.GetInputSourceOneHot()
      gradient_mask = {
          k: tf.tensordot(v, onehot, 1)
          for k, v in self._per_input_gradient_mask.items()
      }
    all_losses = []
    for optimization in self.learners:
      learner_name = optimization.params.name
      (losses, train_ops['train/%s' % learner_name],
       eval_metrics) = optimization.Apply(
           metrics,
           vmap,
           gradient_mask=gradient_mask,
           gradient_adjuster=self.AdjustGradients)
      all_losses.extend(losses)
      if add_summary:
        for key, (value, weight) in eval_metrics.items():
          self.AddEvalMetric(
              key + '/' + learner_name,
              value,
              weight,
              raise_if_already_added=not py_utils.IsEagerMode())

    relevant_bn_updates, _ = py_utils.FindRelevantBatchNormUpdates(
        all_losses, tf.get_collection(py_utils.BATCH_NORM_UPDATES))
    train_ops['bn_updates'] = relevant_bn_updates

    var_update_ops = [
        tf.group(*tf.nest.flatten(train_ops), name='var_update_ops')
    ]
    # Post training step update.
    with tf.control_dependencies(var_update_ops):
      post_step_op = self.PostTrainingStepUpdate()

    train_ops = {}
    with tf.control_dependencies([post_step_op]):
      # Get the op to update the weight masks and thresholds
      mask_update_op = self._GetMaskUpdateOp()
      train_ops['mask_updates'] = mask_update_op
      with tf.control_dependencies([mask_update_op]):
        true_global_step = py_utils.GetOrCreateGlobalStepVar()
        with tf.ops.colocate_with(true_global_step):
          if self.params.defer_global_step_update:
            increment_global_steps = true_global_step
          else:
            increment_global_steps = tf.assign_add(true_global_step, 1)
        # TF2 will treat (tensor1 != tensor2) as a boolean tensor, so avoid
        # using inequality here.
        if self._global_step_var is not true_global_step:
          with tf.ops.colocate_with(self._global_step_var):
            increment_global_steps = tf.group(
                increment_global_steps, tf.assign_add(self._global_step_var, 1))
        train_ops['global_step'] = increment_global_steps

    if not py_utils.IsEagerMode():
      # Some of the values could be a tf.no_op(), which returns None in eager
      # mode, so we don't want to check that when eager is enabled.
      for op_name, op in train_ops.items():
        if op is None:
          raise ValueError(f'Train op {op_name} is None.')
    return train_ops

  def _BPropForVariables(self, vmap):
    """Constructs the backward graph."""
    train_ops = self._BPropGenTrainOps(vmap)

    # TODO(rpang): try to structure _train_op as:
    #   tf.cond(skip_step, <only update skip stats>, <all updates>)
    # so that we skip all other updates when a step is skipped.
    with tf.control_dependencies(
        [tf.group(*tf.nest.flatten(train_ops), name='train_ops')]):
      self._train_op = tf.group(self._post_train_ops, name='bprop')

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
          tf.logging.info('Keep gradient after filtering, regex: %s var: %s' %
                          (bprop_variable_filters[i], var.name))
          self._per_input_gradient_mask[var.name] += (
              tf.one_hot(i, len(bprop_variable_filters), dtype=tf.float32))

  def ApplyExponentialMovingAverage(self, ema):
    """Wraps `self.train_op` with an op updating exponential moving average."""
    if (self._create_variables_status !=
        base_layer._CreateLayerVariablesStatus.COMPLETED):  # pylint: disable=protected-access
      raise ValueError(
          'ApplyExponentialMovingAverage called before InstantiateVariables!')
    # TODO(rpang): raise an exception if this is called in the eval mode.
    all_vars = _VariablesForEMA(self.params, self.vars.Flatten())
    with tf.name_scope('moving_average'):
      self._post_train_ops.append(ema.apply(all_vars))

  # TODO(blee): Rename Decode->DecodeWithDefaultTheta, DecodeWithTheta->Decode.
  def Decode(self, input_batch):
    """Constructs the inference graph for eval decoding.

    Args:
      input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
        spiltting is used, a list of `NestedMap`, one for each split.

    Returns:
      a dict of Tensors as decoder output.
    """
    return self.DecodeWithTheta(self.theta, input_batch)

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph for eval decoding with theta.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
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

  def DecodeFinalize(self, decode_finalize_args):
    """Finalize any work for decoding.

    Args:
      decode_finalize_args: A DecodeFinalizeArgs namedtuple.
    """
    decode_out_path = decode_finalize_args.decode_out_path
    decode_out = decode_finalize_args.decode_out
    if decode_out:
      decoder_lib.WriteKeyValuePairs(decode_out_path, decode_out)

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
  def post_training_loop_op(self):
    assert self._post_training_loop_op is not None, (
        'No post_training_loop_op op is defined. Call PostTrainingLoop first.')
    return self._post_training_loop_op

  @property
  def global_step(self):
    return self._global_step_var

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

  def AddEvalMetric(self, name, value, weight, raise_if_already_added=True):
    """Adds a metric to the eval metrics.

    Args:
      name: A python string. The name of the metric.
      value: A scalar Tensor.
      weight: A scalar Tensor.
      raise_if_already_added: If the metric already exists, raise a ValueError.

    Raises:
      ValueError: if `name` is already defined.

    """
    if name in self._eval_metrics and not tf.executing_eagerly():
      if raise_if_already_added:
        raise ValueError('Metric %s has already been defined.' % name)
    self._eval_metrics[name] = (value, weight)

  def AddPerExampleTensor(self, name, value):
    if name in self._per_example and not tf.executing_eagerly(
    ) and not py_utils.IsEagerMode():
      raise ValueError('Metric %s has already been defined.' % name)
    self._per_example[name] = value

  def _UpdateVnConfig(self):
    """Update vn config from the various vn flags."""
    p = self.params
    tp = p.train
    if tp:
      vn_enabled = ((tp.vn_std > 0) and p.vn and
                    (p.vn.global_vn or p.vn.per_step_vn))
      if self.do_eval or (not vn_enabled):
        p.vn = py_utils.VariationalNoiseParams(None, False, False)
      else:
        # vn.scale is dependent on global_step.
        if p.vn.scale is not None:
          raise ValueError('A value should not be specified for p.vn.scale. '
                           'It will be overwritten by p.train.vn_std.')
        if p.vn.start_step:
          raise ValueError(
              'A value should not be specified for p.vn.start_step. '
              'It will be overwritten by p.train.vn_start_step.')
        p.vn.scale = tp.vn_std
        p.vn.start_step = tp.vn_start_step

  def _GetMaskUpdateOp(self):
    """Returns op to update masks and threshold variables for model pruning."""
    p = self.params
    tp = p.train
    mask_update_op = tf.no_op()
    if tp.pruning_hparams_dict:
      assert isinstance(tp.pruning_hparams_dict, dict)
      pruning_hparams = pruning.get_pruning_hparams().override_from_dict(
          tp.pruning_hparams_dict)
      if not pruning_utils.UsePruningInterface(tp.pruning_hparams_dict):
        pruning_obj = pruning.Pruning(
            pruning_hparams, global_step=py_utils.GetGlobalStep())
        if self.cluster.add_summary:
          pruning_obj.add_pruning_summaries()
        mask_update_op = pruning_obj.conditional_mask_update_op()

      if (pruning_utils.UsePruningInterface(tp.pruning_hparams_dict) and
          pruning_utils.PruningOp.ApplyTensorflowUpdate()):
        mask_update_op = pruning_utils.PruningOp.GetPruningUpdate()
    return mask_update_op

  def Export(self, train_dir):
    """Called by an eval job before evaluation.

    Can be used to write additional information to disk.

    Args:
      train_dir: Directory in which any additional files should be saved. This
        is also the same directory where checkpoints will be written.
    """
    pass


class BaseModel(base_layer.BaseLayer):
  """The abstract model class. All models are sub-class of this class."""

  @classmethod
  def Params(cls):
    p = super().Params()
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
    tp.Define('tpu_steps_per_loop', 1000, 'The number of training steps per '
              'training loop for TPUs.')
    tp.Define(
        'tpu_device_order_mode', None,
        'A device_assignment_lib.DeviceOrderMode enum that determines whether '
        'to assign devices in a way that the order of replicas or '
        'model-parallel cores will form a ring or mesh, or let the library to '
        'choose. Default None to AUTO.')
    tp.Define(
        'tpu_computation_shape', None,
        'A 4-element list that describes how virtual cores (which we specify '
        'in TF computation) should be mapped to one or more logical cores.')
    tp.Define(
        'ema_decay', 0.0,
        'If > 0, enable ExponentialMovingAverage during training '
        'with the give decay. '
        'Must be < 1. Disabled if <= 0. '
        'Must be set consistent across all tasks.')
    tp.Define(
        'ema_decay_moving_vars', None,
        'If True, include variables from collection "moving_vars" in ema. '
        'Must be set consistent across all tasks.')
    tp.Define('init_from_checkpoint_rules', {},
              'See BaseTask documentation for details.')
    tp.Define('init_from_checkpoint_override', '',
              'See BaseTask document for details.')
    tp.Define('early_stop', None,
              'Early stopping based on dev-set performance.')
    tp.Define(
        'enqueue_max_steps', -1, 'Max enqueue steps. -1 meaning no limit.'
        ' This flag should be set for unit-test only.')
    tp.Define('save_interval_seconds', 60 * 10,
              'Generates a checkpoint roughly once every this many seconds.')
    tp.Define(
        'save_interval_steps', None,
        'Generates a checkpoint roughly once every this many training '
        'steps. Supersedes save_interval_seconds if not None.')
    tp.Define('save_max_to_keep', 100,
              'Maximum number of recent checkpoints to keep.')
    tp.Define('save_keep_checkpoint_every_n_hours', 0.5,
              'How often to keep a checkpoint.')
    tp.Define('summary_interval_steps', 100,
              'Generates a checkpoint roughly once every this many steps.')
    tp.Define('async_checkpointing', False,
              'Checkpointing asynchronously. Currently only support executor.')
    return p

  def __init__(self, params):
    """Initializes this Model."""
    assert issubclass(params.cls, BaseModel)
    super().__init__(params)
    tf.logging.info('Training parameters for %s: %s', params.cls,
                    self.params.train)
    self._global_step_var = py_utils.GetOrCreateGlobalStepVar()

    tp = self.params.train
    if tp.ema_decay > 0:
      assert tp.ema_decay < 1.0
      # Use the global EMA if set (for multi-task training).
      self._ema = py_utils.ExponentialMovingAverage()
      if not self._ema:
        self._ema = tf.train.ExponentialMovingAverage(
            decay=tp.ema_decay, num_updates=self.global_step)
    else:
      assert not py_utils.ExponentialMovingAverage()
      self._ema = None

  @property
  def global_step(self):
    return self._global_step_var

  @property
  def ema(self):
    return self._ema

  @property
  def variables_for_ema(self):
    return _VariablesForEMA(self.params, self.vars.Flatten())

  def ConstructFPropBPropGraph(self):
    raise NotImplementedError('Abstract method')

  def ConstructFPropGraph(self):
    raise NotImplementedError('Abstract method')

  def ConstructPostTrainingLoop(self, outfeed=None):
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

  def ProcessFPropResults(self, sess, global_step, metrics, per_example):
    """Called once for each train loop.

    BaseTask.ProcessFPropResults is also called on each loop, so you
    can put your implementation wherever it is most convenient for you.

    Be sure to implement BaseTask.FilterPerExampleTensors if you plan to use any
    per-example tensors in this method.

    Args:
      sess: a session.
      global_step: model global step. Since ProcessFPropResults is called after
        sess.run(train_op), this value will be p.train.tpu_steps_per_loop higher
        than the value in FProp.
      metrics: the metrics dict returned by FPropTower.
      per_example: the per_example dict returned by FPropTower.
    """
    pass

  def Export(self, train_dir):
    """Called by an eval job after evaluation.

    Can be used to write additional information to CNS.

    Args:
      train_dir: Directory in which any additional files should be saved. This
        is also the same directory where checkpoints will be written.
    """
    for task in self.tasks:
      task.Export(train_dir)


class SingleTaskBase(BaseModel):
  """Represents a single task from a model.

  Subclasses must create a Task in self._task by the end of __init__.
  """

  @property
  def tasks(self):
    return [self._task]

  def GetTask(self, task_name=None):
    assert not task_name, 'Must not specify >task_name< for single-task model.'
    return self._task

  def SampleTask(self, global_step):
    return self._task

  def ConstructFPropBPropGraph(self):
    if self.ema:
      tf.logging.info('ApplyExponentialMovingAverage on %s', self._task)
      self._task.ApplyExponentialMovingAverage(self.ema)
    self._task.FPropDefaultTheta()
    self._task.BProp()

  def ConstructFPropGraph(self):
    self._task.FPropDefaultTheta()

  def ConstructPostTrainingLoop(self, outfeed=None):
    self._task.PostTrainingLoop(outfeed)


class SingleTaskModel(SingleTaskBase):
  """Model that consists of a single task."""

  @classmethod
  def Params(cls, task_params=None):
    p = super().Params()
    p.Define(
        'task', None,
        '`InstantiableParams` object for a `BaseTask` or its derivatives.')

    if task_params is not None:
      # Copy over model parameters from the task parameters.
      p.task = task_params
      base_layer.BaseLayer.CopyBaseParams(p.task, p)
      tp = p.train
      tp.start_up_delay_steps = p.task.train.start_up_delay_steps
      tp.max_steps = p.task.train.max_steps
      tp.tpu_steps_per_loop = p.task.train.tpu_steps_per_loop
      tp.tpu_device_order_mode = p.task.train.tpu_device_order_mode
      tp.tpu_computation_shape = p.task.train.tpu_computation_shape
      # init_from_checkpoint_rules does not need to be copied.
      tp.early_stop = p.task.train.early_stop
      tp.enqueue_max_steps = p.task.train.enqueue_max_steps
      tp.save_interval_seconds = p.task.train.save_interval_seconds
      tp.save_interval_steps = p.task.train.save_interval_steps
      tp.save_max_to_keep = p.task.train.save_max_to_keep
      tp.save_keep_checkpoint_every_n_hours = (
          p.task.train.save_keep_checkpoint_every_n_hours)
      tp.summary_interval_steps = p.task.train.summary_interval_steps
      tp.async_checkpointing = p.task.train.async_checkpointing

    return p

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
      if not p.task.input:
        tf.logging.warning('Model input generator is not defined')
      p.input = p.task.input
    p.train.ema_decay = p.task.train.ema_decay
    p.train.ema_decay_moving_vars = p.task.train.ema_decay_moving_vars

    super().__init__(p)
    self.CreateChild('_task', self.params.task)

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    self._task.InstantiateVariables()
    super()._CreateChildrenVariables()


class MultiTaskSubModel(SingleTaskBase):
  """'Model' consisting of a task from a multi-task model.

  The entire multi-task model is constructed, but otherwise this model
  appears to be a SingleTaskModel consisting of just one of the multi-task
  model's tasks.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'multi_task_sub_model'
    p.Define('task_name', '', 'The name of the task to execute from the '
             'enclosing model.')
    return p

  def __init__(self, params, shared_model=None):
    super().__init__(params)
    p = self.params
    self._model = shared_model
    self._task = self._model.children.Get(p.task_name)


class MultiTaskModel(BaseModel):
  """Model that consists of multiple tasks."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'task_params', hyperparams.Params(),
        'Params object mapping task name to `BaskTask`(or derivatives) '
        'Params.')
    p.Define(
        'task_probs', hyperparams.Params(),
        'Params object mapping task name to the relative likelihood the '
        'task will be sampled during training.')
    p.Define('task_schedule', None, 'Task schedule.')
    p.Define(
        'task_global_step', False,
        'Whether or not to use task-specific global steps, which causes each '
        'task to use its own global_step instead of the true global_step. '
        'NOTE: this may be severely broken. Verify carefully!')
    p.Define(
        'task_name_var_scope', True,
        'Whether or not to use the task name as a variable scope. Note that '
        'this has been the default behavior for some time, but seems to be '
        'redundant since the individual tasks scope by their `name`.')
    p.Define(
        'share_model_object', False,
        'If true, during training we create the model object once and share '
        'it between all tasks.')
    return p

  @staticmethod
  def TaskNames(params):
    return sorted(task_name for task_name, _ in params.task_params.IterParams())

  def __init__(self, params):
    assert issubclass(params.cls, MultiTaskModel)
    super().__init__(params)
    p = self.params
    assert len(p.task_params) > 1

    sorted_task_params = sorted(
        (task_name, task_params)
        for task_name, task_params in p.task_params.IterParams())

    # Pass input params to tasks.
    assert isinstance(p.input, hyperparams.Params)

    for task_name, task_params in sorted_task_params:
      assert isinstance(task_params, hyperparams.Params)
      assert not task_params.input
      try:
        task_params.input = p.input.Get(task_name)
      except AttributeError as e:
        tf.logging.error(
            'Missing input params for task %s !'
            'Check that you have the correct datasets '
            'passed to DefineMultitaskDatasets.', task_name)
        raise e

      if p.task_global_step:
        assert task_name == task_params.name, (task_name, task_params.name)
        task_params.task_global_step = True

    assert set(dir(p.input)) == set(dir(p.task_params))

    # For compatibility with older API (with p.task_probs)
    if p.task_schedule is None:
      p.task_schedule = task_scheduler.ConstantScheduler.Params()
      p.task_schedule.task_probs = sorted(list(p.task_probs.IterParams()))

    if p.train.ema_decay > 0:
      for task_name, task_params in sorted_task_params:
        for field in ['ema_decay', 'ema_decay_moving_vars']:
          if task_params.train.Get(field) != p.train.Get(field):
            raise ValueError('Params did not match for field %s in task %s' %
                             (field, task_name))

    # CreateChild copies over global configs in p to individual task params,
    # which then gets propagated down to all sub-layers during
    # BaseTask._PropagateDownGlobalConfigs(), or through sub-sequent CreateChild
    # or CreateChildren calls.
    with tf.name_scope(p.name):
      for task_name, task_params in sorted_task_params:
        self.CreateChild(task_name, task_params)

      self.CreateChild('task_schedule', p.task_schedule)

  def _CreateChildrenVariables(self):
    with tf.name_scope(self.params.name):
      for task_name, task in zip(self.task_names, self.tasks):
        if self.params.task_name_var_scope:
          with tf.variable_scope(task_name):
            task.InstantiateVariables()
        else:
          task.InstantiateVariables()
      self.task_schedule.InstantiateVariables()
    super()._CreateChildrenVariables()

  @property
  def task_names(self):
    return MultiTaskModel.TaskNames(self.params)

  @property
  def tasks(self):
    return [self.children[name] for name in self.task_names]

  def GetTask(self, task_name):
    assert task_name, 'Must specify >task_name< for multi-task model.'
    return self.children[task_name]

  def SampleTask(self, global_step):
    """Returns a sampled task according to self.task_schedule.

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
        if self.ema:
          task.ApplyExponentialMovingAverage(self.ema)
        task.FPropDefaultTheta()
        task.BProp()

  def ConstructFPropGraph(self):
    for task_name in self.task_names:
      with tf.name_scope(task_name):
        task = self.GetTask(task_name)
        task.FPropDefaultTheta()
