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
"""BERT masked language model configurations."""

from typing import List

import jax
from lingvo.jax import base_model_params
from lingvo.jax import layers
from lingvo.jax import model_registry
from lingvo.jax import py_utils
from lingvo.jax.tasks.lm import input_generator
from lingvo.jax.tasks.lm import model_params

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit


class BertDataset(base_model_params.BaseModelParams):
  """MLPerf Bert dataset."""
  PERCORE_BATCH_SIZE = 8
  MLPERF_REMASK = True
  RANDOM_BUFFER_SIZE = 100_000

  def _datasetTrain(self) -> InstantiableParams:
    """Parameters for using the original ML Perf training data."""
    p = input_generator.TFRecordBertInput.Params()
    p.name = 'train'
    p.input_file = 'gs://mlperf_v1_1/bert/train'
    p.enable_packing = True
    num_local_devices = jax.local_device_count()
    p.batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    p.remask = self.MLPERF_REMASK
    p.mlm_augmenter.Set(mask_token_id=103, vocab_size=30522)
    p.num_samples = 156_725_653
    p.file_buffer_size = self.RANDOM_BUFFER_SIZE
    p.is_training = True
    return p

  def _datasetTest(self) -> InstantiableParams:
    """Parameters for using the original ML Perf eval data."""
    p = input_generator.TFRecordBertInput.Params()
    p.name = 'test'
    p.input_file = 'gs://mlperf_v1_1/bert/eval'
    p.enable_packing = False
    p.reset_for_eval = True
    num_local_devices = jax.local_device_count()
    p.batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    p.num_samples = 10_000
    p.is_training = False
    return p

  def datasets(self) -> List[InstantiableParams]:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    """Returns a list of dataset parameters."""
    return [self._datasetTrain(), self._datasetTest()]

  def task(self) -> InstantiableParams:
    raise NotImplementedError()


@model_registry.register_model
class BertAdamL4H128(model_params.TransformerBertPmapAdam, BertDataset):
  """4-layer Transformer LM using Adam on JF 2x2.

  global batch size = 2 * 2 * 2 * 8 = 64
  """
  NUM_LAYERS = 4
  VOCAB_SIZE = 32000
  NUM_HEADS = 8
  MODEL_DIMS = 128
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ACTIVATION_FUNCTION = 'RELU'
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000

  ENABLE_BFLOAT16 = True


class BertSpmd(model_params.TransformerBertSpmdAdafactor, BertDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ENABLE_BFLOAT16 = False
  MASK_TOKEN_ID = 0

  ACTIVATION_FUNCTION = 'RELU'

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 500
  CHECKPOINT_SAVE_MAX_TO_KEEP = 10


@model_registry.register_model
class BertSpmdL4H128(BertSpmd):
  """SPMD model on JF 2x2."""
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 4
  MODEL_DIMS = 128
  DIMS_PER_HEAD = 16
  HIDDEN_DIMS = MODEL_DIMS * 4
  assert MODEL_DIMS % DIMS_PER_HEAD == 0
  NUM_HEADS = int(MODEL_DIMS / DIMS_PER_HEAD)
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  USE_REPEATED_LAYER = False
  ENABLE_BFLOAT16 = True

  # Sub-class has to specify a mesh.
  MESH_SHAPE = [1, 4, 2]

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000


@model_registry.register_model
class BertSpmdL33H12kBiggerBatch(BertSpmd):
  """100B model using 2k global batch size and 512 chips.

  Global batch size = 8 * 8 * 8 * 4 = 2048
  """  # pylint: disable=line-too-long
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 33

  MODEL_DIMS = 12288
  DIMS_PER_HEAD = 96
  HIDDEN_DIMS = MODEL_DIMS * 8
  assert MODEL_DIMS % DIMS_PER_HEAD == 0
  NUM_HEADS = int(MODEL_DIMS / DIMS_PER_HEAD)
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  WEIGHT_DECAY = 1e-2
  USE_REPEATED_LAYER = False
  ENABLE_BFLOAT16 = True
  USE_MLPERF_DATA = True
  MLPERF_REMASK = True
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM

  MESH_SHAPE = [1, 64, 8]

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 1000

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = super().task()
    # Enable label smoothing.
    model_p.label_smoothing_prob = 0.1
    return model_p


@model_registry.register_model
class BertSpmdL33H12kBiggerBatch8x8x16(BertSpmdL33H12kBiggerBatch):
  """100B model using 4k global batch size and 1024 chips.

  Global batch size = 8 * 8 * 16 * 4 = 4096
  """  # pylint: disable=line-too-long
  PERCORE_BATCH_SIZE = 4
  RANDOM_BUFFER_SIZE = 50_000
  MESH_SHAPE = [1, 64, 16]
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 500


@model_registry.register_model
class BertSpmdL66H12kBiggerBatch8x8x16(BertSpmdL33H12kBiggerBatch):
  """200B model using 4k global batch size and 1024 chips.

  Global batch size = 8 * 8 * 16 * 4 = 4096
  """  # pylint: disable=line-too-long
  NUM_LAYERS = 66

  PERCORE_BATCH_SIZE = 4
  RANDOM_BUFFER_SIZE = 50_000
  MESH_SHAPE = [1, 64, 16]
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 200
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20


@model_registry.register_model
class BertSpmdL66H12kBiggerBatch8x8x8(BertSpmdL66H12kBiggerBatch8x8x16):
  r"""200B model using 512 global batch size and 1024 chips.

  Global batch size = 8 * 8 * 8 * 1 = 512
  """  # pylint: disable=line-too-long
  PERCORE_BATCH_SIZE = 1
  MESH_SHAPE = [1, 64, 8]
