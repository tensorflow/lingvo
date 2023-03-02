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
import math
from typing import List, Union
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
  """

  # Note: there are other optimizers implemented by the API, but these two are
  #   the only ones currently needed for now.
  optimizer: Union[TPUEmbeddingAdagradOptimizer, TPUEmbeddingAdamOptimizer]
  schedule: schedule_lib.BaseSchedule

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
  def input_keys(self) -> List[str]:
    return self._input_keys

  @property
  def table_config(self):
    return self._table_config

  @property
  def max_sequence_length(self) -> int:
    return self._max_sequence_length


class TPUEmbeddingLayer(base_layer.BaseLayer):
  """Lingvo interface to TF's TPUEmbedding API."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('num_tpu_hosts', 0, 'Total number of TPU hosts.')
    p.Define('tables', None, 'List[TPUEmbeddingTable]')
    p.Define(
        'pipeline_execution_with_tensor_core',
        False,
        'Set to True to be faster. See tpu_embedding.py for details.',
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
