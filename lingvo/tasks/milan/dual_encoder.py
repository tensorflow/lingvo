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
"""Template for a multi-modal dual encoder model."""

from absl import logging
from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import hyperparams
from lingvo.core import layers as lingvo_layers
from lingvo.core import metrics as metrics_lib
from lingvo.core import py_utils

from lingvo.tasks.milan import labels as label_lib
from lingvo.tasks.milan import score_functions
from lingvo.tasks.milan import tpu_utils
from lingvo.tasks.milan import utils


def _EncodeBatch(encoder: base_layer.BaseLayer, inputs,
                 batch_shape: tf.TensorShape):
  """Runs `encoder` on `inputs` with an arbitrary number of batch dimensions."""
  # Flatten inputs so they have a single batch dimension, run the encoder,
  # and restore the batch dims of the output.
  batch_adapter = utils.BatchFlattener(batch_shape)
  return batch_adapter.Unflatten(encoder(*batch_adapter.Flatten(inputs)))


def EncoderConfig() -> hyperparams.Params:
  """Returns Params for configuring one `DualEncoder` modality."""
  p = hyperparams.Params()
  p.Define(
      'input_features', '',
      'Feature(s) from the input batch to feed to the encoder. The structure '
      'of this field determines the number and structure of the encoder '
      'arguments. Examples: If set to "feature_name", the encoder is called '
      'with a single argument `input_batch["feature_name"]`; if set to an '
      'N-element tuple, it is called with N arguments. See `Selector` class '
      'for more details.')
  p.Define('id_feature', '', 'Name of id feature to use for loss masking.')
  p.Define(
      'encoder', None,
      'Params of a layer that encodes input_features. The layer should '
      'accept the output of Selector(input_features) as arguments.')
  p.Define('output_dim', None,
           'Dimension of the embeddings produced by `encoder`.')
  p.Define('encoder_scope', '',
           'Optional variable scope name to create the encoder in.')
  p.Define('projection_scope', '',
           'Optional variable scope in which to create the projection layer.')
  return p


class DualEncoder(base_layer.BaseLayer):
  """Implements a dual encoder trained with in-batch softmax loss."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('encoder_configs', {}, 'Modality name => EncoderConfig params.')
    p.Define('score_function', score_functions.DotProductScoreFunction.Params(),
             'Layer that computes similarity score.')
    p.Define(
        'joint_embedding_dim', 0,
        'Dimension to project x and y encoders\' outputs. Defaults to '
        'max(x_encoder.output_dim, y_encoder.output_dim).')
    p.Define('regularization_loss_weight', 1.0,
             'Weight of regularization loss.')
    p.Define(
        'loss_weights', {},
        'Weights of retrieval losses. Keys should be modality pair tuples and '
        'values should be floats >= 0.')
    p.Define(
        'label_fn', None, 'Label function to call to label batches during '
        'training. Has signature ExamplePairs -> tf.Tensor.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.name

    if not p.label_fn:
      raise ValueError('Required param label_fn not set.')

    if not p.loss_weights:
      raise ValueError('Required param loss_weights not set.')
    for modality_pair in p.loss_weights:
      if (not isinstance(modality_pair, tuple) or len(modality_pair) != 2 or
          not isinstance(modality_pair[0], str) or
          not isinstance(modality_pair[1], str)):
        raise ValueError(
            'loss_weights: Keys should be tuples of modality names; got '
            f'{modality_pair}')
      for modality in modality_pair:
        if modality not in p.encoder_configs:
          raise ValueError(f'loss_weights: Unknown modality "{modality}"')

    for modality_name, config in p.encoder_configs.items():
      for param_name in ('input_features', 'id_feature'):
        if not getattr(config, param_name):
          raise ValueError(f'Required param {param_name} not set')

      # Set default scope names if not configured.
      if not config.encoder_scope:
        config.encoder_scope = f'{modality_name}_encoder'
      if not config.projection_scope:
        config.projection_scope = f'{modality_name}_projection'
      if not config.encoder.name:
        config.encoder.name = config.encoder_scope

    # Infer joint_embedding_dim if not provided.
    if not p.joint_embedding_dim:
      p.joint_embedding_dim = max(
          config.output_dim for config in p.encoder_configs.values())
      tf.logging.info('Defaulting DualEncoder joint_embedding_dim to %d',
                      p.joint_embedding_dim)

    for name, config in p.encoder_configs.items():
      self.CreateChild(f'encoder_{name}', config.encoder)

    # Where necessary, create layers to project the encoders' output to the
    # joint space.
    base_projection_params = lingvo_layers.ProjectionLayer.Params().Set(
        output_dim=p.joint_embedding_dim,
        activation='NONE',
        has_bias=True,
        batch_norm=False)

    projections = {}
    for modality, config in p.encoder_configs.items():
      if config.output_dim != p.joint_embedding_dim:
        projections[modality] = (
            base_projection_params.Copy().Set(
                name=config.projection_scope, input_dim=config.output_dim))
    self.CreateChildren('projections', projections)

    self.CreateChild('score_function', p.score_function)

  def _child_variable_scope_override(self):
    p = self.params
    res = super()._child_variable_scope_override()
    for modality, config in p.encoder_configs.items():
      res[f'encoder_{modality}'] = [p.name, config.encoder_scope]
    return res

  def EncodeModality(self, modality: str, inputs, batch_shape=None):
    """Runs `inputs` through `modality`'s encoder and optional projection.

    Args:
      modality: Name of the modality to encode.
      inputs: Tensor(s) of input to the encoder, e.g. a batch of decoded images.
      batch_shape: TensorShape describing the batch structure of the inputs.
        Defaults to `[None]`, which means `inputs` have a single batch
        dimension. Set to (e.g.) `[None, 5]` if each example in the batch
        contains 5 encodable items.

    Returns:
      A float32 Tensor of the encoded items, shape
      `batch_shape + [joint_embedding_dim]`
    """
    if batch_shape is None:
      batch_shape = tf.TensorShape([None])
    if not isinstance(inputs, tuple):
      inputs = (inputs,)

    encodings = _EncodeBatch(
        self.children[f'encoder_{modality}'], inputs, batch_shape=batch_shape)

    # If necessary, project outputs to joint_embedding_dim.
    if modality in self.projections:
      return self.projections[modality](encodings)
    else:
      return encodings

  # NB: ComputePredictions and ComputeLoss methods below mimic the interface of
  # a BaseTask.

  def ComputePredictions(self, theta, input_batch):
    """Encodes the examples in `input_batch` with respect to each modality.

    Args:
      theta: `NestedMap` containing variable values of this task.
      input_batch: `NestedMap` of input tensors.

    Returns:
      A `NestedMap` of encodings in all configured modalities. Maps modality
      name to a `NestedMap` with

        - 'ids': int32 ids Tensor, shape `[batch_size, ...]`
        - 'encodings': float32 encodings Tensor, shape
          `ids_shape + [joint_embedding_dim]`
    """
    del theta  # Unused
    p = self.params
    input_batch = py_utils.NestedMap(input_batch)

    outputs = py_utils.NestedMap()
    for modality, config in p.encoder_configs.items():
      inputs = utils.Selector(config.input_features)(input_batch)
      if not isinstance(inputs, tuple):
        inputs = (inputs,)
      ids = utils.GetFromNestedMapOrDie(input_batch, config.id_feature)
      outputs[modality] = py_utils.NestedMap(
          encodings=self.EncodeModality(
              modality, inputs, batch_shape=ids.shape),
          ids=ids)

    return outputs

  def ComputeLoss(self, theta, predictions, input_batch):
    """Computes loss and other metrics for the given predictions.

    Args:
      theta: A `.NestedMap` object containing variable values of this task.
      predictions: The output of `ComputePredictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A tuple (metrics, per_example_tensors), where
        - `metrics` is a dict of str keys to (metric, weight) values
        - `per_example_tensors` is a dict of str keys to tensors describing each
          training example, where the first dimension of each tensor is the
          batch index.
    """
    p = self.params

    # During TPU training, collect the encodings and ids from all TPUs so the
    # loss can be computed over all query-result pairs in the global batch.
    # To avoid duplicating work, each TPU operates on a non-overlapping
    # slice of these pairs. Specifically, each TPU uses queries drawn from its
    # local batch and results from the global batch.

    # Encodings of the local and global examples, keyed by modality.
    local_flat_encodings = py_utils.NestedMap({
        modality: tf.reshape(predictions[modality].encodings,
                             [-1, p.joint_embedding_dim])
        for modality in predictions
    })
    global_flat_encodings = tpu_utils.ConcatenateAcrossReplicas(
        local_flat_encodings)

    def _ComputePerQueryLoss(query_modality, result_modality):
      labeler_inputs = label_lib.ExamplePairs.BetweenLocalAndGlobalBatches(
          input_batch,
          query_modality=query_modality,
          result_modality=result_modality)
      labels = p.label_fn(labeler_inputs)

      # [num_queries, num_results]
      flat_similarities = self.score_function(
          local_flat_encodings[query_modality],
          global_flat_encodings[result_modality])

      flat_labels = tf.reshape(labels, flat_similarities.shape)
      # [num_queries]
      return label_lib.MultiLabelContrastiveLoss(
          labels=flat_labels, logits=flat_similarities)

    loss_terms = []
    metrics = {}
    for direction, loss_weight in p.loss_weights.items():
      query_modality, result_modality = direction
      if not loss_weight:
        logging.info('Skipping %s retrieval', direction)
        continue
      per_query_losses = _ComputePerQueryLoss(query_modality, result_modality)
      mean_per_query_loss = tf.reduce_mean(per_query_losses)
      loss_terms.append(loss_weight * mean_per_query_loss)
      metrics['loss_{}_to_{}'.format(query_modality,
                                     result_modality)] = (mean_per_query_loss,
                                                          1)

    regularization_losses = utils.CollectRegularizationLosses(self)
    if p.regularization_loss_weight and regularization_losses:
      tf.logging.info('Adding TF1 regularization loss: %s',
                      regularization_losses)
      total_reg_loss = tf.reduce_sum(regularization_losses)
      loss_terms.append(p.regularization_loss_weight * total_reg_loss)
      metrics['loss_regularization'] = (total_reg_loss, 1)

    loss = tf.add_n(loss_terms)
    metrics['loss'] = (loss, 1)
    return metrics, {}


class MilanTask(base_model.BaseTask):
  """Task that runs a `DualEncoder`."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('dual_encoder', DualEncoder.Params(),
             'Configuration of the DualEncoder to train.')
    p.name = 'milan'
    return p

  def __init__(self, params):
    if not params.name:
      raise ValueError('params.name not set.')
    super().__init__(params)
    p = self.params
    # Construct the model.
    self.CreateChild('dual_encoder', p.dual_encoder)

  def ComputePredictions(self, theta, input_batch):
    return self.dual_encoder.ComputePredictions(theta.dual_encoder, input_batch)

  def ComputeLoss(self, theta, predictions, input_batch):
    return self.dual_encoder.ComputeLoss(theta.dual_encoder, predictions,
                                         input_batch)

  # Methods below implement parts of `BaseTask` that get called by lingvo
  # 'decoder' jobs. The current minimal implementation just causes summaries to
  # be generated.

  def Decode(self, input_batch):
    preds = self.ComputePredictions(self.theta, input_batch)
    # Add summary ops to the graph.
    _ = self.ComputeLoss(self.theta, preds, input_batch)
    return preds

  def CreateDecoderMetrics(self):
    return {
        'num_samples_in_batch': metrics_lib.AverageMetric(),
    }
