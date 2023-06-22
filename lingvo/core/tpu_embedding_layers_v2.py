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

from typing import Callable, Optional, Union, Sequence

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import schedule as schedule_lib
from lingvo.core import tpu_embedding_layers
from lingvo.core import tpu_embedding_manager

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_embedding_v2_utils
# pylint:enable=g-direct-tensorflow-import

TPU_EMBEDDING_MANAGER: tpu_embedding_manager.TPUEmbeddingManager = None
"""The global singleton TPUEmbeddingManager instance.

This is used by the host driven executor and its associated TrainProgram.
"""


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
    self.tpu_embedding_manager: tpu_embedding_manager.TPUEmbeddingManager = (
        TPU_EMBEDDING_MANAGER
    )

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
      TPU_EMBEDDING_MANAGER.InitializeMidlevelApi(
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
      tf.logging.info(
          'Running build() on %s', TPU_EMBEDDING_MANAGER.tpu_embedding
      )
      TPU_EMBEDDING_MANAGER.tpu_embedding.build()

      # Keep the manager as an attribute to ensure the underlying API object is
      #   included in model serialization.
      self.tpu_embedding_manager = TPU_EMBEDDING_MANAGER
      tf.logging.info(
          'Configured TPUEmbedingManager: %s',
          TPU_EMBEDDING_MANAGER,
      )

  def _TpuEmbLookup(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """See base class."""
    return self.tpu_embedding_manager.Lookup(ids_map)
