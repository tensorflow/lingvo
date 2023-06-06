# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Manages the mid-level API.

Note the global singleton of this class is a module property of
tpu_embedding_layers_v2.py.
"""

from typing import AbstractSet, Optional, Tuple, Mapping, MutableSequence, Sequence

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import schedule as schedule_lib

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
# pylint:enable=g-direct-tensorflow-import

TPUEmbedding = tpu_embedding_v2.TPUEmbedding
EmbeddingFeature = tf.tpu.experimental.HardwareFeature.EmbeddingFeature


def _DenseToSparse(dense_tensor: tf.Tensor) -> tf.sparse.SparseTensor:
  """Like tf.sparse.from_dense but uses -1 as its padding value."""
  indices = tf.where(tf.not_equal(dense_tensor, -1))
  return tf.sparse.SparseTensor(
      indices=tf.cast(indices, tf.int64),
      values=tf.gather_nd(dense_tensor, indices),
      dense_shape=dense_tensor.shape,
  )


class TPUEmbeddingManager(tf.autotrackable.AutoTrackable):
  """Manages a global singleton instance of tpu_embedding_v2.TPUEmbedding."""

  def __init__(self):
    self.reset()

  def InitializeMidlevelApi(
      self,
      feature_config: py_utils.NestedMap,
      optimizer: Optional[tpu_embedding_v2_utils._Optimizer] = None,
      pipeline_execution_with_tensor_core: bool = True,
  ):
    """Initializes the TF embedding mid-level API object."""
    tf.logging.info('[TPUEmbeddingManager]: Init midlevel api')

    # Initialize the correct tpu embedding API for the hardware we have.
    strategy = tf.distribute.get_strategy()
    tpu_emb_feature = strategy.extended.tpu_hardware_feature.embedding_feature
    if tpu_emb_feature == EmbeddingFeature.V1:
      self._tpu_embedding = tpu_embedding_v2.TPUEmbedding(
          feature_config, optimizer, pipeline_execution_with_tensor_core
      )
    else:
      raise ValueError('TPUEmbeddingManager is only compatible with V1')

  def reset(self):
    """Resets object state.

    In particular, removes the reference to the TPUEmbedding object, if present,
    necessary for re-running unit tests with a fresh API object.
    """

    # True when a TPUEmbeddingLayer has initialized this class. Otherwise, all
    # operations pass through. This eliminates the need to conditionally check
    # everywhere in the host driven executor code whether the client model is
    # using this API.
    self.enabled = False
    self._tpu_embedding: TPUEmbedding = None
    self._feature_names: AbstractSet[str] = frozenset()
    self._gradient_multiplier_schedule: schedule_lib.BaseSchedule = None
    self._summary_tensors: MutableSequence[Tuple[str, tf.Tensor, tf.Tensor]] = (
        []
    )
    self._sequence_features: AbstractSet[str] = frozenset()

    # Used to cache activations upon each Dequeue, for later Lookup calls.
    self._activations: py_utils.NestedMap = py_utils.NestedMap()

  def __bool__(self):
    """Passes through the indicator expressing whether the v2 API is enabled."""
    return self.enabled

  def __repr__(self):
    return (
        f'{self.__class__.__name__}: [enabled={self.enabled}],'
        f' [features={self._feature_names}],'
        f' [sequence_features={self._sequence_features}]'
    )

  @property
  def tpu_embedding(self) -> TPUEmbedding:
    """The global singleton TPUEmbedding object used by all TPUEmbeddingLayerV2s."""
    return self._tpu_embedding

  @property
  def feature_names(self) -> AbstractSet[str]:
    """Global singleton set of feature names used across the embedding tables."""
    return self._feature_names

  @feature_names.setter
  def feature_names(self, feature_names: AbstractSet[str]):
    """Sets `feature_names`, the set of all features with embeddings."""
    self._feature_names = frozenset(feature_names)

  @property
  def sequence_features(self) -> AbstractSet[str]:
    """Set of all features with sequence embeddings."""
    return self._sequence_features

  @property
  def non_sequence_features(self) -> AbstractSet[str]:
    """Set of all features with non-sequence embeddings."""
    return self._feature_names - self._sequence_features

  @sequence_features.setter
  def sequence_features(self, features: AbstractSet[str]):
    """Sets `sequence_features`, the set of all sequence features."""
    self._sequence_features = frozenset(features)

  @property
  def gradient_multiplier_schedule(self) -> schedule_lib.BaseSchedule:
    return self._gradient_multiplier_schedule

  @gradient_multiplier_schedule.setter
  def gradient_multiplier_schedule(self, schedule: schedule_lib.BaseSchedule):
    self._gradient_multiplier_schedule = schedule

  def AddSummaryTensor(self, name: str, value: tf.Tensor, weight: float = 1.0):
    self._summary_tensors.append((name, value, tf.convert_to_tensor(weight)))

  @property
  def summary_tensors(self) -> Sequence[Tuple[str, tf.Tensor, tf.Tensor]]:
    """Returns a list of (name, value, weight) tuples for summary."""
    return self._summary_tensors

  def ProcessInputFeature(self, key: str, tensor: tf.Tensor) -> tf.Tensor:
    """Processes a single NestedMap entry for compatibility with the API.

    This function is meant to be called after GetPreprocessedInputBatch but
    importantly *before* its tensors are split across accelerator devices, once
    on each entry in the batch's NestedMap.

    For sequence features, we add an extra dimension of size 1. See
    documentation below in TPUEmbeddingLayer:_CreateLayerVariables.

    For non-sequence features, we convert the id tensors to SparseTensors when
    there are multiple ids per example. We do this to match the logic in our V1
    layers, which always enqueue such ids for lookup as sparse EnqueueData, and
    in so doing always permit the combiner reduction to occur (since this
    mirrors tf.nn.embedding_lookup_sparse's API).

    Args:
      key: the (possibly-nested) key of the NestedMap.
      tensor: the tensor we wish to optionally process.

    Returns:
      A possibly-modified tensor which now conforms to Lingvo's use of the
      TPUEmbedding API. If unmodified, returns the tensor which was passed in.
    """
    if not self.enabled:
      return tensor

    # Add an empty dimension to indicate these are for sequence embeddings.
    if key in self.sequence_features:
      return tf.expand_dims(tensor, -1)

    # Update non-sequence features which have multiple ids per example to
    # sparse tensors, in order to trigger the TPUEmbedding library's
    # reduction logic. This is how the v1 layers behave.
    elif key in self.non_sequence_features and tensor.shape[1] != 1:
      # Sparse conversion only appears necessary when there are multiple IDs
      # per example. There is a performance penalty in enqueing a SparseTensor
      # over a regular dense tensor, so we avoid the conversion when just a
      # single ID is present per example.
      return _DenseToSparse(tensor)

    return tensor

  def SliceEmbeddingIds(
      self, input_batch: py_utils.NestedMap
  ) -> py_utils.NestedMap:
    """Slices out only the embedding ids from an input batch."""
    return input_batch.GetSlice(self.feature_names)

  def _NonSequenceExpandDims(self, key, tensor):
    # The V1 layers automatically add an extra 'time' dimension to non-sequence
    # embedding lookup results. While not ideal, we replicate that pattern
    # here for consistency between V1/V2.
    if key in self.non_sequence_features:
      return tf.expand_dims(tensor, 1)
    return tensor

  def Enqueue(self, batch: py_utils.NestedMap) -> None:
    """Enqueues embedding column ids from input batch to TPU coprocessor.

    Args:
      batch: the input batch containing the id features to enqueue.
    """
    self._activations = py_utils.NestedMap()

    if self.enabled:
      self.tpu_embedding.enqueue(self.SliceEmbeddingIds(batch))

    # Somehow even after TPUEmbeddingLayer sets up the manager, it's getting
    # to this line. No reset in between.

  def Dequeue(self, batch: py_utils.NestedMap) -> py_utils.NestedMap:
    """Dequeues embedding column ids from TPU coprocessor.

    This should be called once per training step, and it dequeues all embedding
    tables. For later table lookups, use Lookup below.

    Args:
      batch: Input batch containing the id columns; added to support future
        subclass features.

    Returns:
      A NestedMap of embedding activations.
    """
    del batch
    if self.enabled:
      self._activations = self.tpu_embedding.dequeue()

      # Note: we watch the activations as they're dequeued - so without the
      # expand_dims call applied (see Lookup). Hence, when applying gradient
      # updates, use expand_nonsequence_features=False to skip that step.
      py_utils.CurrentGradientTape().watch(self._activations)

    return self._activations

  def Lookup(
      self,
      ids: Optional[py_utils.NestedMap] = None,
      expand_nonsequence_features: bool = True,
  ) -> py_utils.NestedMap:
    """Returns the TPUEmbedding activations corresponding to the requested ids.

    TPUEmbedding v1 only permits all activations to be dequeued at once, so we
    cache the output from Dequeue and permit sliced lookups here.

    Args:
      ids: a NestedMap of ids whose corresponding activations are returned.
      expand_nonsequence_features: if False, skips calling expand_dims on non0-
        sequence features. Only do this when computing gradients.
    """
    if not self.enabled:
      return self._activations.copy()

    if ids:
      outputs = self._activations.GetSlice(
          {*ids.Keys()} & {*self._activations.Keys()}
      )
    else:
      outputs = self._activations.copy()

    if expand_nonsequence_features:
      return outputs.TransformWithKey(self._NonSequenceExpandDims)
    return outputs

  def ApplyGradients(
      self, gradients: py_utils.NestedMap
  ) -> Mapping[str, tf.Tensor]:
    """Applies schedule-scaled gradient updates to the embedding variables.

    Args:
      gradients: grads to apply to the embedding variables.

    Returns:
      An eval_metrics dictionary for reporting.
    """
    if not self.enabled:
      return {}

    multiplier = self.gradient_multiplier_schedule.Value()
    scaled_grads = gradients.Transform(lambda g: g * multiplier)

    self.tpu_embedding.apply_gradients(scaled_grads)

    return {
        'tpu_embedding_activation_norm': (
            tf.sqrt(py_utils.SumSquared(self.Lookup().Flatten())),
            tf.constant(1.0),
        ),
        'tpu_embedding_grad_norm': (
            tf.sqrt(py_utils.SumSquared(scaled_grads.Flatten())),
            tf.constant(1.0),
        ),
        'tpu_embedding_gradient_multiplier': (multiplier, tf.constant(1.0)),
    }
