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

"""Common interface authors may use to access Lingvo TPUEmbedding functionality.

Currently, tpu_embedding_layers_v1.py only works with *graph mode, TrainerTpu*
setups, and tpu_embedding_layers_v2.py only with *eager mode,
HostDrivenExecutor* setups. (The reason for this is that the latter depends on
using TPUStrategy to produce a host-driven training loop, and the former is
incompatible with eager mode.)

This file provides common logic and configuration params for each of these two
implementations, so that clients may more freely switch between the APIs.
"""
import abc
import math
from typing import Callable, Sequence
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import schedule as schedule_lib


class TPUEmbeddingOptimizerBase(base_layer.BaseLayer):
  """Base class for TPUEmbeddingLayer & TPUEmbeddingTable optimizers."""

  @classmethod
  def Params(
      cls,
  ) -> hyperparams.InstantiableParams['TPUEmbeddingOptimizerBase']:
    p = super().Params()
    p.Define(
        'clip_weight_min',
        None,
        'The minimum value to clip the weight by; None means -infinity.',
    )
    p.Define(
        'clip_weight_max',
        None,
        'The maximum value to clip the weight by; None means +infinity.',
    )
    p.Define(
        'clip_gradient_min',
        None,
        'The minimum value to clip the gradient by; None means -infinity.',
    )
    p.Define(
        'clip_gradient_max',
        None,
        'The maximum value to clip the gradient by; None means +infinity.',
    )
    p.Define(
        'weight_decay_factor',
        None,
        (
            'Amount of weight decay to apply; None means that the weights are'
            ' not decayed.'
        ),
    )
    p.Define(
        'multiply_weight_decay_factor_by_learning_rate',
        False,
        (
            'If true, weight_decay_factor is multiplied by the current learning'
            ' rate.'
        ),
    )
    p.Define(
        'use_gradient_accumulation',
        True,
        (
            'Setting this to False makes embedding gradients calculation less'
            ' accurate but faster. See tpu_embedding_lib for more details.'
        ),
    )
    return p


class TPUEmbeddingSGDOptimizer(TPUEmbeddingOptimizerBase):
  """SGD optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""


class TPUEmbeddingAdagradOptimizer(TPUEmbeddingOptimizerBase):
  """Adagrad optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'initial_accumulator', 0.1, 'Initial value of Adagrad accumulator.'
    )
    return p


class TPUEmbeddingAdamOptimizer(TPUEmbeddingOptimizerBase):
  """Adam optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'lazy_adam',
        True,
        'Use lazy Adam instead of Adam. Lazy Adam trains faster.',
    )
    p.Define(
        'beta1', 0.9, 'The exponential decay rate for the 1st moment estimates'
    )
    p.Define(
        'beta2',
        0.999,
        'The exponential decay rate for the 2nd moment estimates',
    )
    p.Define('epsilon', 1e-08, 'A small constant for numerical stability')
    p.Define(
        'sum_inside_sqrt',
        True,
        (
            'When this is true, the Adam update formula is changed from'
            ' m / (sqrt(v) + epsilon) to m / sqrt(v + epsilon**2). This option'
            ' improves the performance of TPU training and is not expected to'
            ' harm model quality.'
        ),
    )
    return p


class TPUEmbeddingFTRLOptimizer(TPUEmbeddingOptimizerBase):
  """FTRL optimizer for TPUEmbeddingLayer & TPUEmbeddingTable."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'learning_rate_power',
        -0.5,
        (
            'A float value, must be less or equal to zero. Controls how'
            ' the learning rate decreases during training. Use zero for a fixed'
            ' learning rate.'
        ),
    )
    p.Define(
        'initial_accumulator_value',
        0.1,
        (
            'The starting value for accumulators. Only zero or positive values'
            ' are allowed.'
        ),
    )
    p.Define(
        'l1_regularization_strength',
        0.0,
        'A float value, must be greaterthan or equal to zero. Defaults to 0.0.',
    )
    p.Define(
        'l2_regularization_strength',
        0.0,
        'A float value, must be greaterthan or equal to zero. Defaults to 0.0.',
    )
    p.Define(
        'multiply_linear_by_learning_rate',
        False,
        'Whether multiplylinear by learning rate.',
    )
    p.Define(
        'beta',
        0.0,
        (
            'A float value, representing the beta value from the FTLR paper.'
            ' Defaults to 0.0.'
        ),
    )
    p.Define(
        'allow_zero_accumulator', False, 'Whether allowing zeroaccumulator.'
    )
    p.Define('initial_linear_value', 0.0, 'Initial linear value.')

    return p


class TPUEmbeddingTable(base_layer.BaseLayer):
  """An embedding table controlled by TPUEmbeddingLayer.

  Note that all input_keys need to be declared upfront.

  Attributes:
    optimizer: An optimizer wrapper class around the API object.
    schedule: A Lingvo schedule object used for scaling the learning rate.
  """

  # Type annotations for Lingvo child layers  # TODO(b/275392925) restore these.
  # optimizer: TPUEmbeddingOptimizerBase
  # schedule: schedule_lib.BaseSchedule

  @classmethod
  def Params(cls) -> py_utils.InstantiableParams['TPUEmbeddingTable']:
    p = super().Params()
    p.Define('vocab_size', 0, 'Depth of the input.')
    p.Define('embedding_dim', 0, 'Depth of the output.')
    p.Define('input_keys', None, 'Name of inputs in InputBatch.')
    p.Define(
        'combiner',
        'sum',
        (
            'Must be "sum", "sqrtn", "mean". In the V1 API, use `None` to'
            ' indicate that a feature be treated as a sequence embedding. In'
            ' the V2 API, set it to "sum" (it is ignored).'
        ),
    )
    p.Define(
        'max_sequence_length',
        None,
        (
            'If not None or 0, embedding lookup will return a '
            '"sequence embedding" of shape '
            '`[batch, max_sequence_length, embedding_dim]` without applying a '
            'sequence reducing combiner over dim 1.'
        ),
    )
    p.Define('num_tpu_hosts', 0, 'Total number of TPU hosts.')
    p.Define(
        'optimizer',
        None,
        (
            'Table optimizer parameters. Will override the optimizer '
            "parameters defined in this table's TPUEmbeddingLayer."
        ),
    )
    p.Define(
        'learning_rate', None, "The learning rate for this table's optimizer."
    )
    p.Define('lr_schedule', None, "Overrides TPUEmbeddingLayer's lr_schedule.")
    p.Define(
        'inference_use_merged_variable',
        False,
        (
            'Whether to use merged embedding table variable during inference. '
            'If set to True, only one table variable will be created, and '
            'the user will need to manually merge the sharded table variables '
            'in the trained checkpoint before generating the inference graph.'
        ),
    )
    p.Define(
        'inference_variable_dtype',
        None,
        (
            'The dtype of embedding table variables during inference. If None, '
            'self.params.dtype will be used. Note that the variables in the '
            'inference checkpoint must be with this dtype, and any conversion '
            'from float32 (if necessary) needs to be done separately.'
        ),
    )
    p.Define(
        'inference_auxiliary_variable_specs',
        None,
        (
            'A dict of variable_name -> (dtype, shape) for any auxiliary '
            'variables that the layer need to create during inference. For '
            'example, if quantization techniques are used, it may need to '
            'record the value range (i.e. min/max) of the table variables.'
        ),
    )
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dim > 0
    assert p.input_keys
    assert p.name
    assert p.num_tpu_hosts > 0
    assert p.optimizer
    assert p.learning_rate
    assert p.lr_schedule

    self._ids_per_shard = int(math.ceil(float(p.vocab_size) / p.num_tpu_hosts))
    self._padded_vocab_size = self._ids_per_shard * p.num_tpu_hosts
    self._input_keys = list(p.input_keys)

    self._max_sequence_length = 0
    if p.max_sequence_length:
      self._max_sequence_length = p.max_sequence_length

    self.CreateChild('optimizer', p.optimizer)
    self.CreateChild('schedule', p.lr_schedule)
    self._table_name = f'{p.name}_table'

  @property
  def table_name(self) -> str:
    return self._table_name

  @property
  def input_keys(self) -> Sequence[str]:
    return self._input_keys

  @property
  def table_config(self):
    return self._table_config

  @property
  def max_sequence_length(self) -> int:
    return self._max_sequence_length

  def _SequenceEmbLookup(
      self, dense_ids: tf.Tensor, partition_strategy: str
  ) -> tf.Tensor:
    """Performs embedding lookup for a sequence embedding (no reduction).

    Note that we do not support padding ids in sequence embeddings.

    Args:
      dense_ids: An int Tensor of shape [batch, sequence].
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      A float32 activations Tensor of shape
      [batch, max_sequence_length, embedding_dim].
    """
    p = self.params
    embeddings = tf.nn.embedding_lookup(
        params=self.theta.wm,
        ids=tf.reshape(dense_ids, [-1]),
        partition_strategy=partition_strategy,
    )
    out_shape = tf.concat([tf.shape(dense_ids), [p.embedding_dim]], 0)
    return tf.reshape(embeddings, out_shape)

  def _CombinerEmbLookup(
      self, sparse_ids: tf.SparseTensor, partition_strategy: str
  ) -> tf.Tensor:
    """Performs embedding lookup for a sequence embedding (with a reduction).

    Args:
      sparse_ids: An int SparseTensor of shape [batch, ...].
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      A float32 activations Tensor of shape [batch, 1, embedding_dim].
    """
    p = self.params
    embeddings = tf.nn.embedding_lookup_sparse(
        self.theta.wm,
        sp_ids=sparse_ids,
        sp_weights=None,
        combiner=p.combiner,
        partition_strategy=partition_strategy,
    )
    batch_size = sparse_ids.dense_shape[0]
    # For tf.nn.embedding_lookup_sparse, output.dim0 might be different from
    # sparse_ids.dense_shape.dim0.
    # Explicitly pad results to maintain dim0=batch.
    dim0_padlen = tf.cast(batch_size, tf.int32) - tf.shape(embeddings)[0]
    embeddings = tf.pad(embeddings, [[0, dim0_padlen], [0, 0]])
    # [batch, 1, embedding_dim]
    embeddings = py_utils.HasShape(embeddings, [batch_size], ndims=1)
    return tf.expand_dims(embeddings, 1)

  def CpuEmbLookup(
      self, ids_map: py_utils.NestedMap, partition_strategy: str
  ) -> py_utils.NestedMap:
    """Fetch embedding values for the given ids.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, sequence] int
        Tensor. For sequence embeddings, -1 is used as a padding id.
        Non-sequence embeddings do not support padded ids.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations NestedMap of nested string -> float32 Tensor.
      For non-sequence embeddings: [batch, 1, embedding_dim]
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
    """
    # "Sequence embedding": no combiner case.
    if self.max_sequence_length > 0:
      return ids_map.Transform(
          lambda ids: self._SequenceEmbLookup(ids, partition_strategy)
      )

    # Non-"Sequence embedding", combiner case
    def _Lookup(ids):
      # Dense to sparse.
      dense_shape = tf.shape(ids, out_type=tf.int64)
      sample_indices = tf.cast(tf.where(tf.not_equal(ids, -1)), tf.int64)
      embedding_indices = tf.cast(tf.gather_nd(ids, sample_indices), tf.int64)
      # [?, embedding_dim]
      sparse_ids = tf.SparseTensor(
          indices=sample_indices,
          values=embedding_indices,
          dense_shape=dense_shape,
      )
      return self._CombinerEmbLookup(sparse_ids, partition_strategy)

    return ids_map.Transform(_Lookup)

  def CpuEmbLookupSparse(
      self, ids_map: py_utils.NestedMap, partition_strategy: str
  ) -> py_utils.NestedMap:
    """Fetch embedding values for the given sparse ids.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, ...] int
        SparseTensor.
      partition_strategy: See TPUEmbeddingLayer partition_strategy param.

    Returns:
      An activations NestedMap of nested string -> float32 Tensor.
      For non-sequence embeddings: [batch, 1, embedding_dim]
      For sequence embeddings: [batch, max_sequence_length, embedding_dim]
    """
    if self.max_sequence_length > 0:
      # "Sequence embedding", no combiner case
      def _Lookup(ids):
        return self._SequenceEmbLookup(
            tf.sparse.to_dense(ids, default_value=-1), partition_strategy
        )

      return ids_map.Transform(_Lookup)

    # Non-"Sequence embedding", combiner case
    return ids_map.Transform(
        lambda ids: self._CombinerEmbLookup(ids, partition_strategy)
    )


class TPUEmbeddingLayer(
    base_layer.BaseLayer, metaclass=base_layer.ABCLayerMeta
):
  """Lingvo interface to TF's TPUEmbedding API.

  Attributes:
    tables: The TPUEmbeddingTables this layer manages.
    gradient_multiplier_schedule: A Lingvo schedule object used for scaling the
      gradients applied to the embedding variables.
  """

  # Type annotations for Lingvo child layers  # TODO(b/275392925) restore these.
  # tables: Sequence[TPUEmbeddingTable]
  # gradient_multiplier_schedule: schedule_lib.BaseSchedule

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_tpu_hosts', 0, 'Total number of TPU hosts.')
    p.Define('tables', None, 'Sequence[TPUEmbeddingTable]')
    p.Define(
        'pipeline_execution_with_tensor_core',
        False,
        (
            'Enabling this option will speed training up at some quality cost.'
            ' For larger embeddings, the cost is relatively smaller, because it'
            " is more likely that step n-1's updates don't affect step n's"
        ),
    )
    p.Define('batch_size', 0, 'Per-core batch size.')
    p.Define(
        'optimizer',
        TPUEmbeddingAdagradOptimizer.Params(),
        (
            'Layer optimizer parameters. Will be used for any'
            ' TPUEmbeddingTables with None optimizer parameters.'
        ),
    )
    p.Define('learning_rate', 0.0, 'Learning rate.')
    p.Define(
        'lr_schedule',
        schedule_lib.ContinuousSchedule.Params(),
        'Lingvo learning rate schedule. Will be multiplied to learning rate.',
    )
    p.Define(
        'partition_strategy',
        'div',
        (
            'A string, either "mod" or "div", specifying how to map the lookup'
            ' id to the embedding tensor. For more information see'
            ' `tf.nn.embedding_lookup_sparse`.'
        ),
    )
    p.Define(
        'gradient_multiplier_schedule',
        schedule_lib.ConstantOne.Params(),
        (
            'Values from this schedule will be multiplied to the embedding'
            ' gradients. Gradients from non-TPU Embedding variables will not be'
            ' affected.'
        ),
    )
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    assert p.tables
    assert p.batch_size > 0
    assert p.name
    assert p.gradient_multiplier_schedule
    assert p.partition_strategy in ('mod', 'div')

    if p.num_tpu_hosts > 0:
      for table_params in p.tables:
        num_tpu_hosts = table_params.num_tpu_hosts
        if 0 < num_tpu_hosts != p.num_tpu_hosts:
          raise ValueError(f'{num_tpu_hosts=} != {p.num_tpu_hosts=}')
        table_params.num_tpu_hosts = p.num_tpu_hosts
    else:
      num_tpu_hosts = p.tables[0].num_tpu_hosts
      assert all(t.num_tpu_hosts == num_tpu_hosts for t in p.tables)

    # Stop if both a table and the layer have no optimizer related parameters.
    for param_name in ('optimizer', 'learning_rate', 'lr_schedule'):
      table_param_missing = any(
          table_params.Get(param_name) is None for table_params in p.tables
      )
      if not p.Get(param_name) and table_param_missing:
        raise ValueError(
            f'A table is missing {param_name} parameters, and no layer-level '
            f'{param_name} parameters were given.'
        )
      elif table_param_missing:
        for table_params in p.tables:
          if table_params.Get(param_name) is None:
            value = p.Get(param_name)
            if isinstance(value, hyperparams.Params):
              value = value.Copy()  # Avoid mutating the original copy.
            table_params.Set(**{param_name: value})

    self.CreateChildren('tables', p.tables)
    self.CreateChild(
        'gradient_multiplier_schedule', p.gradient_multiplier_schedule
    )

  @abc.abstractmethod
  def _TpuEmbLookup(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """Use the TPU Embedding API to perform the lookup. Varies by API."""

  def _CheckIdsMap(self, ids_map: py_utils.NestedMap) -> None:
    """Check that the keys in `ids_map` is valid for embedding lookup."""
    assert isinstance(ids_map, py_utils.NestedMap)
    valid_keys = set()
    for table in self.tables:
      for input_key in table.input_keys:
        valid_keys.add(input_key)
    invalid_keys = set(ids_map.Keys()) - valid_keys
    if invalid_keys:
      raise ValueError(
          f'Invalid input keys: {invalid_keys}. (Valid keys: {valid_keys})'
      )

  def _LookupImpl(
      self,
      ids_map: py_utils.NestedMap,
      method_selector: Callable[
          [TPUEmbeddingTable],
          Callable[[py_utils.NestedMap, str], py_utils.NestedMap],
      ],
  ) -> py_utils.NestedMap:
    """Implementation of common lookup logic; dispatches between CPU/TPU.

    Args:
      ids_map: The ids to fetch activations for.
      method_selector: a callable that takes a table and returns the method to
        be used for the lookup.

    Returns:
      A NestedMap of activations either computed on CPU or fetched from the TPU.
    """
    self._CheckIdsMap(ids_map)
    p = self.params

    # TPU Lookup
    if py_utils.IsTpuTraining(p):
      return self._TpuEmbLookup(ids_map)

    # CPU Lookup
    ret = py_utils.NestedMap()
    for table in self.tables:
      ret.Update(
          method_selector(table)(
              ids_map.GetSlice({*table.input_keys} & {*ids_map.Keys()}),
              p.partition_strategy,
          )
      )
    return ret

  def EmbLookup(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """Looks up embedding vectors for each entry in dense Tensor ids_map.

    Since the TPUEmbedding is monolithic, and consulted once per FProp/BProp, we
    must centralize the lookup. Thus, for multiple features, we contain them
    into a single-lookup rather than allowing the caller to call Lookup multiple
    times.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, sequence] int
        Tensor. For sequence embeddings, -1 is used as a padding id.
        Non-sequence embeddings do not support padded ids.

    Returns:
      Activations NestedMap of nested string ->
        For non-sequence embeddings:  [batch, 1, embedding_dim],
        For sequence embeddings: [batch, max_sequence_length, embedding_dim]
        float32 Tensor.
    """
    return self._LookupImpl(ids_map, lambda table: table.CpuEmbLookup)

  def EmbLookupSparse(self, ids_map: py_utils.NestedMap) -> py_utils.NestedMap:
    """Looks up embedding vectors for each entry in SparseTensor ids_map.

    Since the TPUEmbedding is monolithic, and consulted once per FProp/BProp, we
    must centralize the lookup. Thus, for multiple features, we contain them
    into a single-lookup rather than allowing the caller to call Lookup multiple
    times.

    Args:
      ids_map: A NestedMap of nested `input_key` string -> [batch, ...] int
        SparseTensor.

    Returns:
      Activations NestedMap of nested string ->
        For non-sequence embeddings:  [batch, 1, embedding_dim],
        For sequence embeddings: [batch, max_sequence_length, embedding_dim]
        float32 Tensor.
    """
    return self._LookupImpl(ids_map, lambda table: table.CpuEmbLookupSparse)
