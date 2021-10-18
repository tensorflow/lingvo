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
"""Base class for all Jax models."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jax.experimental import pjit
from lingvo.jax import base_layer
from lingvo.jax import layers
from lingvo.jax import metric_utils
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax import train_states
import optax
import tensorflow.compat.v2 as tf

NestedMap = py_utils.NestedMap
JTensor = base_layer.JTensor
NestedJTensor = base_layer.NestedJTensor
JTensorOrPartitionSpec = base_layer.JTensorOrPartitionSpec
JTensorOrPartitionSpecOrNone = Optional[base_layer.JTensorOrPartitionSpec]
NestedJTensorOrPartitionSpec = base_layer.NestedJTensorOrPartitionSpec
NestedBool = base_layer.NestedBool
NestedParams = base_layer.NestedParams
InstantiableParams = py_utils.InstantiableParams
Predictions = Union[JTensor, NestedMap, Dict[str, Any]]
Metrics = Dict[str, Tuple[JTensor, JTensor]]
PartitionSpec = pjit.PartitionSpec
PyTreeDef = pytypes.PyTreeDef
TrainState = train_states.TrainState

_UNPARTITIONED_BASENAME = 'unpartitioned.msgpack'
_PARTITIONED_SUBDIR = 'partitioned'
_CHECKPOINT_PREFIX = 'ckpt'


class Learner(base_layer.BaseLayer):
  """A learner."""

  @classmethod
  def Params(cls) -> InstantiableParams:
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
    super().__init__(params)
    p = self.params
    assert p.optimizer is not None
    assert p.loss_name is not None
    self._optimizer = p.optimizer.Instantiate()
    self._grad_tx = self._optimizer.GetGradTransformation()

  @property
  def optimizer(self) -> optimizers.BaseOptimizer:
    """Return the Optimizer object of this learner."""
    return self._optimizer

  @property
  def grad_tx(self) -> optax.GradientTransformation:
    return self._grad_tx

  def ScaleGradients(self, grads: NestedMap) -> NestedMap:
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
    base_layer.AddSummary(f'{learner_name}/grad_norm', grad_norm)
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

      def ScaleGradient(grad, norm):
        return grad * jnp.minimum(
            jnp.array(1, grad_norm.dtype),
            jnp.array(p.optimizer.clip_gradient_single_norm_to_value,
                      grad_norm.dtype) / norm)

      grads = jax.tree_map(ScaleGradient, grads, grad_single_norm)
    return grads

  def UpdateStates(self, grads: NestedMap, states: optax.OptState,
                   old_vars: NestedJTensor) -> Tuple[NestedMap, optax.OptState]:
    """Applies gradient transformation, updates optimizer states.

    Args:
      grads: A nested structure of gradient values.
      states: Optimizer states.
      old_vars: Current model weights.

    Returns:
      transformed_grad, new_states pair.
    """
    grads = self.ScaleGradients(grads)
    return self._grad_tx.update(grads, states, old_vars)

  def ApplyGradient(
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

    def _AdjustVar(old_var, transformed_grad, is_learnable):
      if is_learnable:
        return old_var + transformed_grad
      else:
        return old_var

    return tf.nest.map_structure(_AdjustVar, old_vars, transformed_grads,
                                 var_is_learnable)
    # TODO(yonghui): export gradient / variable summaries.

  @property
  def loss_name(self) -> str:
    return self._params.loss_name


class BaseTask(base_layer.BaseLayer):
  """A jax task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('train', py_utils.Params(),
             'Params to control how this task should be trained.')
    tp = p.train
    tp.Define('learner', Learner.Params(), 'One or a list of learners.')
    tp.Define('num_train_steps', 1e7,
              'Maximum number of training steps to run.')
    # TODO(bf-jax): Add an option to perform this wrt. a time duration.
    tp.Define(
        'save_interval_steps', 5000,
        'How frequently to save a model checkpoint in terms of the number of '
        'training steps.')
    tp.Define('save_max_to_keep', 10,
              'The maximum number of recent checkpoints to keep.')
    tp.Define(
        'summary_interval_steps', 100,
        'How frequently to generate summaries in terms of the number of '
        'training steps.')
    tp.Define(
        'norm_summary_interval_steps', 500,
        'How frequently to generate expensive summaries computing the norms '
        'of variables in terms of the number of training steps.')
    tp.Define(
        'eval_interval_steps', 100,
        'How frequently to evaluate the model on the evaluation splits in '
        'terms of the number of training steps.')
    tp.Define(
        'inputs_split_mapping', None, 'The PartitionSpec for inputs'
        'such as inputs, labels, targets, paddings, num words etc. This is only'
        'relevant for SPMD sharded models. By default it is None, which means'
        'all the inputs are replicated. For sharding inputs, this is a '
        '`NestedMap` with keys `map_1d`, `map_2d`, ..., etc.,'
        'which specifies how to shard the inputs of that dimension.')
    tp.Define(
        'decoder_inputs_split_mapping', None, 'The PartitionSpec for decoder'
        'inputs such as partially completed sequence. This is only relevant'
        'for SPMD sharded models. By default it is None, which means all the'
        'inputs are replicated. For sharding the decoder inputs, this is a '
        '`NestedMap` with keys `map_1d`, `map_2d` ..., etc., which specifies'
        'how to shard the decoder inputs corresponding to that dimension.')
    tp.Define(
        'decoder_states_split_mapping', None, 'The PartitionSpec for cached'
        'decoder states such as keys, values, steps etc. This is only relevant'
        'for SPMD sharded models. By default it is None, which means all the'
        'inputs are replicated. For sharding the decoder states, this is a '
        '`NestedMap` with keys `map_1d`, `map_2d` ..., etc., which specifies'
        'how to shard the decoder states corresponding to that dimension.')

    # TODO(yonghui): Add other hyper-params.
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self._params
    assert p.train.learner is not None
    # TODO(yonghui): implement multiple learners.
    assert not isinstance(p.train.learner, (tuple, list))
    learner_params = [p.train.learner]
    self.CreateChildren('learner', learner_params)

  @property
  def learners(self):
    return self.learner

  def ComputePredictions(
      self, theta: NestedMap,
      input_batch: NestedMap) -> Union[JTensor, NestedMap, Dict[str, Any]]:
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
      input_batch: A `.NestedMap` object containing input tensors.

    Returns:
      Predictions, either a single Tensor, a `.NestedMap`, or a namedtuple.
    """
    raise NotImplementedError('Abstract method')

  def ComputeLoss(self, theta: NestedMap, predictions: Union[JTensor,
                                                             NestedMap],
                  input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    raise NotImplementedError('Abstract method')

  def FProp(self, theta: NestedMap,
            input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
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

  def Decode(self, theta: NestedMap, input_batch: NestedMap) -> NestedMap:
    """Decodes input_batch with model weights 'theta'.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: The input batch. A `NestedMap` of tensors. Or, if input batch
        spiltting is used, a list of `NestedMap`, one for each split.

    Returns:
      a dict of Tensors as decoder output.
    """
    raise NotImplementedError('Abstract method')

  def CreateTrainStatePartitionSpecs(self,
                                     var_weight_params) -> Optional[TrainState]:
    """Creates partition specs for all variables used in training.

    Args:
      var_weight_params: a nested map of variable params for all the forward
        variables.

    Returns:
      A TrainState contains PartitionSpecs for all the forward and backward
        variables, or None.
    """
    p = self.params
    device_mesh = p.device_mesh
    if device_mesh is None:
      return None
    else:
      learners = self.learners
      grad_txs = [x.grad_tx for x in learners]
      for grad_tx in grad_txs:
        assert isinstance(grad_tx, optimizers.ShardedGradientTransformation)
      opt_var_weight_params = [
          x.init_partition_spec(var_weight_params) for x in grad_txs
      ]
      var_partition_specs = base_layer.VarPartitionSpecs(
          var_weight_params,
          device_mesh=device_mesh,
          device_axis_names=p.mesh_axis_names)
      opt_var_partition_specs = base_layer.VarPartitionSpecs(
          opt_var_weight_params,
          device_mesh=device_mesh,
          device_axis_names=p.mesh_axis_names)
      step_partition_spec = PartitionSpec()
      return TrainState(
          step=step_partition_spec,
          mdl_vars=var_partition_specs,
          opt_states=opt_var_partition_specs)

  def CreateTrainState(self, mdl_vars: NestedJTensor,
                       var_weight_params: NestedJTensor) -> TrainState:
    """Creates train states that holds all the forward/backward variables.

    Args:
      mdl_vars: A nested structure of model vars to create TrainState for.
        'mdl_vars' can be a sub-set of self.vars.
      var_weight_params: WeightParams for each of the variable in mdl_vars.
        var_weight_params must be of the same structure as mdl_vars. Each model
        weight variable is associated with some WeightParams which contains all
        the meta information about the weight variable.

    Returns:
      a TrainState.
    """
    # Make a private copy of mdl_vars and var_weight_params structures that are
    # not shared with the caller.
    mdl_vars = tf.nest.map_structure(lambda x: x, mdl_vars)
    var_weight_params = tf.nest.map_structure(lambda x: x, var_weight_params)
    learners = self.learners
    grad_txs = [x.grad_tx for x in learners]
    tf.nest.assert_same_structure(mdl_vars, var_weight_params)
    opt_states = [x.init(mdl_vars) for x in grad_txs]
    return TrainState(
        # The global step for the model.
        step=jnp.array(0, dtype=jnp.uint32),
        mdl_vars=mdl_vars,
        opt_states=opt_states)


class LanguageModel(BaseTask):
  """Language Model base task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('lm', layers.TransformerLm.Params(), 'LM layer.')
    p.Define(
        'return_predictions', False, 'Whether to return predictions during'
        'eval. Returning predictions is more expensive, but may be useful'
        'for debugging.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    # Construct the model.
    lm_p = p.lm.Copy()
    self.CreateChild('lm', lm_p)

  def ComputePredictions(self, theta: NestedMap,
                         input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    p = self.params
    inputs = input_batch.ids
    paddings = input_batch.paddings
    labels = NestedMap(
        class_ids=input_batch.labels, class_weights=input_batch.weights)
    if p.lm.packed_input:
      packed_input_kwargs = {
          'segment_ids': input_batch.segment_ids,
          'segment_pos': input_batch.segment_pos,
      }
    else:
      packed_input_kwargs = {}
    return self.lm.FProp(
        theta=theta.lm,
        inputs=inputs,
        paddings=paddings,
        labels=labels,
        **packed_input_kwargs)

  def ComputeLoss(self, theta: NestedMap, predictions: NestedMap,
                  input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    labels = input_batch.labels
    weights = input_batch.weights
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    num_preds = predictions.total_weight
    mean_acc = jnp.sum(
        (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
    metric_weight = jnp.array(1.0, predictions.avg_xent.dtype)
    metrics = py_utils.NestedMap(
        total_loss=(predictions.total_loss, metric_weight),
        avg_xent=(predictions.avg_xent, metric_weight),
        aux_loss=(predictions.aux_loss, metric_weight),
        log_pplx=(predictions.avg_xent, metric_weight),
        fraction_of_correct_next_step_preds=(mean_acc,
                                             jnp.array(1.0, mean_acc.dtype)),
        num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
    )
    per_example_output = py_utils.NestedMap()
    if self.params.return_predictions:
      per_example_output = predictions
    return metrics, per_example_output


class ClassificationTask(BaseTask):
  """Classification task for images and video."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('classifier_params', layers.ResNet.Params(),
             'The classifier network, which is ResNet-50 by default.')
    p.Define('softmax_params', layers.SingleShardFullSoftmax.Params(),
             'The softmax layer used for the classification.')
    p.Define(
        'input_field', 'image',
        'The input field which contains the image or video features to'
        'pass to the classification network.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params

    # Construct the classifier model.
    self.CreateChild('network', p.classifier_params)

    # Construct the softmax layer.
    self.CreateChild('softmax', p.softmax_params)

  def ComputePredictions(self, theta: NestedMap,
                         input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A NestedMap containing str keys and features, softmax output and the
        class weights as values.
    """
    p = self.params
    inputs = input_batch.Get(p.input_field)
    features = self.network.FProp(theta.network, inputs)
    batch_size = inputs.shape[0]
    example_weights = jnp.ones([batch_size])
    if 'weight' in input_batch:
      example_weights = input_batch.weight
      if example_weights.shape != (batch_size,):
        raise ValueError(
            f'Shape of example weights should be ({batch_size},), but instead'
            f'is {example_weights.shape}')
    # Softmax expects weights to be of shape [..., 1].
    softmax_output = self.softmax.FProp(
        theta=theta.softmax,
        inputs=features,
        class_weights=example_weights[:, jnp.newaxis],
        class_probabilities=input_batch.label_probs)
    return NestedMap(
        features=features,
        softmax_output=softmax_output,
        example_weights=example_weights)

  def ComputeLoss(self, theta: NestedMap, predictions: NestedMap,
                  input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to correspond to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index. The base class just returns an empty dict.
    """
    avg_xent = predictions.softmax_output.avg_xent
    total_weight = predictions.softmax_output.total_weight
    metrics = NestedMap(
        avg_xent=(avg_xent, total_weight),
        num_predictions=(total_weight, jnp.array(1.0, total_weight.dtype)))
    if self.do_eval:
      # Compute top-1 and top-5 accuracy and add summary.
      acc1 = metric_utils.top_k_accuracy(
          1,
          predictions.softmax_output.logits,
          label_probs=input_batch.label_probs,
          weights=predictions.example_weights)
      acc5 = metric_utils.top_k_accuracy(
          5,
          predictions.softmax_output.logits,
          label_probs=input_batch.label_probs,
          weights=predictions.example_weights)
      metrics.update(
          accuracy=(acc1, predictions.softmax_output.total_weight),
          acc5=(acc5, predictions.softmax_output.total_weight),
          error=(1.0 - acc1, predictions.softmax_output.total_weight),
          error5=(1.0 - acc5, predictions.softmax_output.total_weight))
      # Add top-1 and top-5 accuracies to summaries.
      base_layer.AddSummary('acc1', acc1)
      base_layer.AddSummary('acc5', acc5)
    return metrics, {}


class BertModel(BaseTask):
  """Bert Model base task."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('lm', layers.TransformerLm.Params(), 'Bert lm layer.')
    p.Define(
        'label_smoothing_prob', 0.0,
        'If > 0.0, smooth out one-hot prob by spreading this amount of'
        ' prob mass to all other tokens.')
    p.Define('mask_token_id', 0, 'Mask token id')

    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.lm.masked_lm
    assert p.lm.packed_input

    self.CreateChild('lm', p.lm)

    mlm_augment_p = layers.MaskedLmDataAugmenter.Params()
    mlm_augment_p.vocab_size = p.lm.vocab_size
    mlm_augment_p.mask_token_id = p.mask_token_id
    self.CreateChild('mlm_augmenter', mlm_augment_p)

  def ComputePredictions(self, theta: NestedMap,
                         input_batch: NestedMap) -> Predictions:
    """Computes predictions for `input_batch`."""
    p = self.params
    assert p.lm.packed_input
    segment_ids = input_batch.segment_ids
    segment_pos = input_batch.segment_pos
    paddings = input_batch.paddings
    # Note that internal BertTransformer uses input_batch.ids instead.
    labels = input_batch.labels
    if 'masked_ids' in input_batch:
      # Input data already has masking done.
      augmented_labels = input_batch.masked_ids
      augmented_pos = input_batch.masked_pos
    else:
      augmented_labels, augmented_pos = self.mlm_augmenter.FProp(
          theta.mlm_augmenter, labels, paddings)

    if p.label_smoothing_prob > 0.0:
      class_probabilities = jax.nn.one_hot(labels, p.lm.vocab_size)
      fill_prob = p.label_smoothing_prob / (p.lm.vocab_size - 1)
      class_probabilities = (
          (1.0 - p.label_smoothing_prob) * class_probabilities + fill_prob *
          (1.0 - class_probabilities)).astype(self.fprop_dtype)

      # Only compute loss on masked pos.
      labels = NestedMap(
          class_probabilities=class_probabilities, class_weights=augmented_pos)
    else:
      # Only compute loss on masked pos.
      labels = NestedMap(class_ids=labels, class_weights=augmented_pos)

    lm_out = self.lm.FProp(
        theta=theta.lm,
        inputs=augmented_labels,
        paddings=paddings,
        labels=labels,
        segment_ids=segment_ids,
        segment_pos=segment_pos)
    lm_out.augmented_labels = augmented_labels
    lm_out.augmented_pos = augmented_pos
    return lm_out

  def ComputeLoss(self, theta: NestedMap, predictions: NestedMap,
                  input_batch: NestedMap) -> Tuple[Metrics, Dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (metric, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    labels = input_batch.labels
    num_tokens = jnp.sum(1.0 - input_batch.paddings.astype(jnp.float32))
    num_seqs = jnp.sum(
        jnp.amax(input_batch.segment_ids.astype(jnp.float32), axis=1))
    weights = predictions.augmented_pos.astype(jnp.float32)
    predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
    num_preds = predictions.total_weight.astype(jnp.float32)
    mean_acc = jnp.sum(
        (labels == predicted_labels) * weights) / jnp.maximum(num_preds, 1)
    metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)
    metrics = py_utils.NestedMap(
        total_loss=(predictions.total_loss, metric_weight),
        avg_xent=(predictions.avg_xent, metric_weight),
        aux_loss=(predictions.aux_loss, metric_weight),
        log_pplx=(predictions.avg_xent, metric_weight),
        fraction_of_correct_preds=(mean_acc, jnp.array(num_preds,
                                                       mean_acc.dtype)),
        num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
        num_tokens=(num_tokens, jnp.array(1.0, num_tokens.dtype)),
        num_seqs=(num_seqs, jnp.array(1.0, num_seqs.dtype)),
    )

    per_example_output = py_utils.NestedMap()
    return metrics, per_example_output
