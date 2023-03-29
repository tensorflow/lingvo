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
"""A lingvo interface into the Tensorflow TPUEmbedding V2 library.

Overview:
  The TPUEmbeddingLayer defined below grants Lingvo model authors access to the
  TensorFlow TPUEmbedding API to train extra-large embedding tables. Similar to
  other Embedding layers in Lingvo, the interface provided is an `EmbLookup`
  function, which may be called from an FProp block to fetch embedding
  activations for particular batches of ids. This library's advantage is that
  it takes advantage of a TPU coprocessor that permits embeddings to be sharded
  across the HBM of large TPU topologies, and thus can vastly increase the total
  size of the embedding tables one can train.

Restrictions:
  Use of this V2 Layer *requires* TPUStrategy, and thus also the Lingvo
  HostDrivenExecutor (and associated Programs).

See also:
  www.tensorflow.org/api_docs/python/tf/tpu/experimental/embedding/TPUEmbedding
"""
import abc
import collections

from typing import Callable, AbstractSet, Optional, Tuple, Union, Mapping, MutableSequence, Sequence

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import schedule as schedule_lib
from lingvo.core import tpu_embedding_layers

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
# pylint:enable=g-direct-tensorflow-import


class _TPUEmbeddingOptimizerV2Mixin(
    base_layer.BaseLayer, metaclass=base_layer.ABCLayerMeta
):
  """Defines the inferface for optimizers expected by the V2 Layer below."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'low_dimensional_packing_status',
        False,
        (
            'Controls whether to optimize 1-, 2-, and 4-dimensional embedding'
            ' tables.'
        ),
    )
    return p

  @abc.abstractmethod
  def CreateOptimizerFn(
      self,
      learning_rate: Union[float, Callable[[], float]],
  ) -> tpu_embedding_v2_utils._Optimizer:  # pylint: disable=protected-access
    """Create TPUEmbedding API optimizer parameters."""


class TPUEmbeddingSGDOptimizer(
    tpu_embedding_layers.TPUEmbeddingSGDOptimizer,
    _TPUEmbeddingOptimizerV2Mixin,
):
  """SGD optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  def CreateOptimizerFn(
      self, learning_rate: Union[float, Callable[[], float]]
  ) -> tpu_embedding_v2_utils.SGD:
    p = self.params
    return tpu_embedding_v2_utils.SGD(
        learning_rate=learning_rate,
        use_gradient_accumulation=p.use_gradient_accumulation,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            p.multiply_weight_decay_factor_by_learning_rate
        ),
        clipvalue=(p.clip_gradient_min, p.clip_gradient_max),
        low_dimensional_packing_status=p.low_dimensional_packing_status,
    )


class TPUEmbeddingAdagradOptimizer(
    tpu_embedding_layers.TPUEmbeddingAdagradOptimizer,
    _TPUEmbeddingOptimizerV2Mixin,
):
  """Adagrad optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  def CreateOptimizerFn(
      self, learning_rate: Union[float, Callable[[], float]]
  ) -> tpu_embedding_v2_utils.Adagrad:
    p = self.params
    return tpu_embedding_v2_utils.Adagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=p.initial_accumulator,
        use_gradient_accumulation=p.use_gradient_accumulation,
        weight_decay_factor=p.weight_decay_factor,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        multiply_weight_decay_factor_by_learning_rate=(
            p.multiply_weight_decay_factor_by_learning_rate
        ),
        clipvalue=(p.clip_gradient_min, p.clip_gradient_max),
        low_dimensional_packing_status=p.low_dimensional_packing_status,
    )


class TPUEmbeddingAdamOptimizer(
    tpu_embedding_layers.TPUEmbeddingAdamOptimizer,
    _TPUEmbeddingOptimizerV2Mixin,
):
  """Adam optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  def CreateOptimizerFn(
      self, learning_rate: Union[float, Callable[[], float]]
  ) -> tpu_embedding_v2_utils.Adam:
    p = self.params
    return tpu_embedding_v2_utils.Adam(
        learning_rate=learning_rate,
        beta_1=p.beta1,
        beta_2=p.beta2,
        epsilon=p.epsilon,
        lazy_adam=p.lazy_adam,
        sum_inside_sqrt=p.sum_inside_sqrt,
        use_gradient_accumulation=p.use_gradient_accumulation,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            p.multiply_weight_decay_factor_by_learning_rate
        ),
        clipvalue=(p.clip_gradient_min, p.clip_gradient_max),
        low_dimensional_packing_status=p.low_dimensional_packing_status,
    )


class TPUEmbeddingFTRLOptimizer(
    tpu_embedding_layers.TPUEmbeddingFTRLOptimizer,
    _TPUEmbeddingOptimizerV2Mixin,
):
  """FTRL optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  def CreateOptimizerFn(
      self, learning_rate: Union[float, Callable[[], float]]
  ) -> tpu_embedding_v2_utils.FTRL:
    p = self.params
    return tpu_embedding_v2_utils.FTRL(
        learning_rate=learning_rate,
        learning_rate_power=p.learning_rate_power,
        l1_regularization_strength=p.l1_regularization_strength,
        l2_regularization_strength=p.l2_regularization_strength,
        beta=p.beta,
        initial_accumulator_value=p.initial_accumulator_value,
        use_gradient_accumulation=p.use_gradient_accumulation,
        clip_weight_min=p.clip_weight_min,
        clip_weight_max=p.clip_weight_max,
        weight_decay_factor=p.weight_decay_factor,
        multiply_weight_decay_factor_by_learning_rate=(
            p.multiply_weight_decay_factor_by_learning_rate
        ),
        slot_variable_creation_fn=None,
        clipvalue=(p.clip_gradient_min, p.clip_gradient_max),
        multiply_linear_by_learning_rate=p.multiply_linear_by_learning_rate,
        allow_zero_accumulator=p.allow_zero_accumulator,
        low_dimensional_packing_status=p.low_dimensional_packing_status,
    )


class TPUEmbeddingTable(tpu_embedding_layers.TPUEmbeddingTable):
  """An embedding table controlled by TPUEmbeddingLayer.

  Note that all input_keys need to be declared upfront.
  """

  optimizer: _TPUEmbeddingOptimizerV2Mixin
  schedule: schedule_lib.BaseSchedule

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    def _LearningRateFn():
      lr = self.schedule.Value() * p.learning_rate
      TPU_EMBEDDING_MANAGER.AddSummaryTensor(f'tpu_embedding_lr/{p.name}', lr)
      return lr

    # pylint: disable=protected-access
    # Create the variable initializer for tpu embedding table.
    create_initializer_fn = (
        py_utils._CreateVarInitStateless
        if py_utils.use_stateless_vars_init()
        else py_utils._CreateVarInitStateful
    )
    initializer = create_initializer_fn(
        name='tpu_embedding_v2',
        method=p.params_init.method,
        shape=[self._padded_vocab_size, p.embedding_dim],
        dim0=self._padded_vocab_size,
        seed=p.params_init.seed,
        scale=p.params_init.scale,
        init_dtype=tf.float32,
    )
    # pylint: enable=protected-access

    # This is the actual TPUEmbedding API object that TPUEmbeddingTable wraps.
    self._table_config = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self._padded_vocab_size,
        dim=p.embedding_dim,
        initializer=initializer,
        optimizer=self.optimizer.CreateOptimizerFn(_LearningRateFn),
        combiner=p.combiner,
        name=f'{self._table_name}_config',
    )

  @property
  def table_config(self) -> tpu_embedding_v2_utils.TableConfig:
    return self._table_config

  def GetDeviceName(self, host_id: int) -> Optional[str]:
    """Return device to place sharded variables on."""
    if self.params.is_inference:
      # This is to place variables on the same device as other variables.
      return None
    if self.do_eval:
      raise NotImplementedError(
          'Pending authorship of host-driven eval program'
      )
    else:
      worker = self.cluster.params.worker.name
      return f'{worker}/replica:0/task:{host_id}/device:CPU:0'


def _DenseToSparse(dense_tensor: tf.Tensor) -> tf.sparse.SparseTensor:
  """Like tf.sparse.from_dense but uses -1 as its padding value."""
  indices = tf.where(tf.not_equal(dense_tensor, -1))
  return tf.sparse.SparseTensor(
      indices=tf.cast(indices, tf.int64),
      values=tf.gather_nd(dense_tensor, indices),
      dense_shape=dense_tensor.shape,
  )


class _TPUEmbeddingManager(tf.autotrackable.AutoTrackable):
  """Manages a global singleton instance of tpu_embedding_v2.TPUEmbedding."""

  def __init__(self):
    self.reset()

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
    self._tpu_embedding: tpu_embedding_v2.TPUEmbedding = None
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

  @property
  def tpu_embedding(self) -> tpu_embedding_v2.TPUEmbedding:
    """The global singleton TPUEmbedding object used by all TPUEmbeddingLayerV2s."""
    return self._tpu_embedding

  @tpu_embedding.setter
  def tpu_embedding(self, tpu_embedding: tpu_embedding_v2.TPUEmbedding):
    if self._tpu_embedding is not None:
      raise ValueError('TPUEmbedding has already been set.')
    self._tpu_embedding = tpu_embedding

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
      self.tpu_embedding.enqueue(batch)

  def Dequeue(self) -> py_utils.NestedMap:
    """Dequeues embedding column ids from TPU coprocessor.

    This should be called once per training step, and it dequeues all embedding
    tables. For later table lookups, use Lookup below.

    Returns:
      A NestedMap of embedding activations.
    """
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


TPU_EMBEDDING_MANAGER = _TPUEmbeddingManager()


class TPUEmbeddingLayer(tpu_embedding_layers.TPUEmbeddingLayer):
  """Interface to TPU embedding which uses the TF2 TPUEmbedding API."""

  # Type annotations for lingvo child objects
  tables: Sequence[TPUEmbeddingTable]
  optimizer: _TPUEmbeddingOptimizerV2Mixin

  @classmethod
  def Params(cls):
    p = super().Params()
    # We override this parameter so that it has a valid default.
    p.optimizer = TPUEmbeddingAdagradOptimizer.Params()
    return p

  def __init__(self, params):
    super().__init__(params)
    self.tpu_embedding_manager: _TPUEmbeddingManager = TPU_EMBEDDING_MANAGER

  def _CreateLayerVariables(self):
    p = self.params

    if not py_utils.IsTpuTraining(p):
      return

    # Note: Several FeatureConfigs can refer to the same TableConfig.
    feature_config = py_utils.NestedMap()
    sequence_features = []
    # There are two ways to indicate a feature should not be combined (and thus
    # treated as a sequence feature):
    # option 1)
    #   Keep inputs as rank 2 sparse tensors and use max_sequence_length -
    #   it may also be useful to call 'embedding.build(per_replica_batch_size)'
    # option 2)  <- We implement this one for now.
    #   Call expand_dims with axis=-1 on sequence tensors (in
    #   `TPUEmbeddingManager:ProcessInputBatch`), making them of rank 3 and
    #    manually set output_shape to [batch_size, max_sequence_length].
    for table in self.tables:
      for feature in table.input_keys:
        # We manually set the output shape here for two reasons:
        #  1. To ensure that sequence features are correctly not reduced, and
        #  2. To ensure that enqueued sparse tensors (for non-sequence features
        #     with >1 id per example) are reduced correctly.
        if table.max_sequence_length > 0:
          output_shape = [p.batch_size, table.max_sequence_length]
          sequence_features.append(feature)
        else:
          output_shape = [p.batch_size]

        feature_config.Set(
            feature,
            tpu_embedding_v2_utils.FeatureConfig(
                table=table.table_config, output_shape=output_shape
            ),
        )

    if not TPU_EMBEDDING_MANAGER:
      TPU_EMBEDDING_MANAGER.enabled = True
      # Note: this line needs to be in a TPUStrategy scope. (We are in one here,
      #   because this function is called indirectly from
      #   HostDrivenExecutor:__init__.)
      # Note: the dequeued activations are packed in the same structure as the
      #   feature_config we provide here (i.e. a NestedMap in Lingvo).
      TPU_EMBEDDING_MANAGER.tpu_embedding = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config,
          optimizer=None,  # Each table will have its own optimizer.
          pipeline_execution_with_tensor_core=(
              p.pipeline_execution_with_tensor_core
          ),
      )
      TPU_EMBEDDING_MANAGER.gradient_multiplier_schedule = (
          self.gradient_multiplier_schedule
      )
      TPU_EMBEDDING_MANAGER.sequence_features = frozenset(sequence_features)

      feature_names = collections.Counter()
      for table in self.tables:
        feature_names.update(table.input_keys)
      if any(v > 1 for v in feature_names.values()):
        raise ValueError(f'Key used by multiple tables ({feature_names=})')
      TPU_EMBEDDING_MANAGER.feature_names = frozenset(feature_names)

      # Build can be done here solely because all output shapes are fully
      # defined in our configuration. This is needed to ensure variables are
      # properly initialized in advance of e.g. checkpoint loading.
      TPU_EMBEDDING_MANAGER.tpu_embedding.build()

      # Keep the manager as an attribute to ensure the underlying API object is
      #   included in model serialization.
      self.tpu_embedding_manager = TPU_EMBEDDING_MANAGER

  def _TpuEmbLookup(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """See base class."""
    return self.tpu_embedding_manager.Lookup(ids_map)
