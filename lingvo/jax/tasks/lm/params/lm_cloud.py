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
"""Decoder-only language model configurations."""

from typing import List

import jax
from jax import numpy as jnp
from lingvo.jax import base_input
from lingvo.jax import base_model_params
from lingvo.jax import layers
from lingvo.jax import model_registry
from lingvo.jax import py_utils
from lingvo.jax.tasks.lm import input_generator
from lingvo.jax.tasks.lm import model_params

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit


class SyntheticDataset(base_model_params.BaseModelParams):
  """Synthetic LM dataset."""
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024

  def _dataset_common(self, is_training) -> InstantiableParams:
    num_local_devices = jax.local_device_count()
    batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    input_p = input_generator.SyntheticLmData.Params()
    if is_training:
      input_p.batch_size = batch_size
    else:
      # TODO(zhangqiaorjc): Is this batch size too big for test?
      input_p.batch_size = batch_size
    input_p.seq_len = self.MAX_SEQ_LEN
    p = base_input.LingvoInputAdaptor.Params().Set(
        input=input_p, is_training=is_training)
    return p

  def datasets(self) -> List[InstantiableParams]:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]


@model_registry.register_model
class LargeMlp(model_params.ClassificationModelAdam, SyntheticDataset):
  """An 8-layer MLP model with large hidden dimensions."""
  NUM_LAYER = 8
  INPUT_DIM = 4096
  OUTPUT_DIM = 4096
  HIDDEN_DIM = 7168

  NUM_DEVICES = 8
  MESH_SHAPE = [8, 1, 1]
  MLP_WEIGHT_SHARDING = [-1, -1]
  SOFTMAX_WEIGHT_SHARDING = [-1, -1]


@model_registry.register_model
class SmallMlp(model_params.ClassificationModelAdam, SyntheticDataset):
  """An 8-layer MLP model with small hidden dimensions."""
  NUM_LAYER = 8
  INPUT_DIM = 1024
  OUTPUT_DIM = 1024
  HIDDEN_DIM = 1024

  NUM_DEVICES = 8
  MESH_SHAPE = [8, 1, 1]
  MLP_WEIGHT_SHARDING = [-1, -1]
  SOFTMAX_WEIGHT_SHARDING = [-1, -1]


## Data parallel training.


@model_registry.register_model
class LmCloudTransformerAdam(model_params.TransformerLmPmapAdam,
                             SyntheticDataset):
  """32-layer Transformer LM using Adam."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3


@model_registry.register_model
class LmCloudTransformerAdamTest(LmCloudTransformerAdam):
  NUM_LAYERS = 2


## SPMD Model parallel training.


class LmCloudSpmd(model_params.TransformerLmSpmdAdafactor, SyntheticDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 10
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION = 'GELU'

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = super().task()
    model_params.set_default_adam(model_p, self.LEARNING_RATE,
                                  self.WEIGHT_DECAY)
    return model_p


@model_registry.register_model
class LmCloudSpmdTest(LmCloudSpmd):
  """SPMD model with small params for local CPU test run.

  Global batch size = 1 * 1 * 1 * 4 = 4
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 2
  MODEL_DIMS = 64
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 8
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 1, 1]


@model_registry.register_model
class LmCloudSpmd2B(LmCloudSpmd):
  """SPMD model with 2B params.

  Global batch size = 2 * 2 * 1 * 32 = 128
  """
  PERCORE_BATCH_SIZE = 32

  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 4, 1]


@model_registry.register_model
class LmCloudSpmd32B(LmCloudSpmd):
  """SPMD model with 32B params.

  Global batch size = 4 * 4 * 4 * 8 = 512
  """
  PERCORE_BATCH_SIZE = 8

  NUM_LAYERS = 40
  MODEL_DIMS = 8192
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_CONTEXT_AND_OUT_PROJ
  MESH_SHAPE = [1, 16, 4]


@model_registry.register_model
class LmCloudSpmd64B(LmCloudSpmd):
  """SPMD model with 64B params.

  Global batch size = 4 * 4 * 8 * 4 = 512
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 51
  MODEL_DIMS = 10240
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_CONTEXT_AND_OUT_PROJ
  MESH_SHAPE = [1, 16, 8]


@model_registry.register_model
class LmCloudSpmd128B(LmCloudSpmd):
  """SPMD model with 128B params.

  Global batch size = 4 * 8 * 8 * 4 = 1024
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 71
  MODEL_DIMS = 12288
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 64, 4]


@model_registry.register_model
class LmCloudSpmd256B(LmCloudSpmd):
  """SPMD model with 256B params.

  Global batch size = 4 * 8 * 8 * 8 = 2048
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 80
  MODEL_DIMS = 16384
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 64, 8]


@model_registry.register_model
class LmCloudSpmd512B(LmCloudSpmd):
  """SPMD model with 512B params.

  Global batch size = 4 * 8 * 8 * 16 = 4096
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 102
  MODEL_DIMS = 20480
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 64, 16]


@model_registry.register_model
class LmCloudSpmd1024B(LmCloudSpmd):
  """SPMD model with 1024B params.

  Global batch size = 2 * 8 * 16 * 16 = 4096
  """
  PERCORE_BATCH_SIZE = 2

  NUM_LAYERS = 142
  MODEL_DIMS = 24576
  HIDDEN_DIMS = MODEL_DIMS * 4

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 256, 8]


class LmCloudSpmdPipeline(model_params.TransformerLmSpmdPipelineAdafactor,
                          SyntheticDataset):
  """Base config for a pipelined SPMD model."""

  NUM_LAYERS = 10
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION = 'GELU'

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  MICROBATCH_SIZE = None
  NUM_STAGES = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = super().task()
    model_params.set_default_adam(model_p, self.LEARNING_RATE,
                                  self.WEIGHT_DECAY)
    return model_p


@model_registry.register_model
class LmCloudSpmdPipeline9B(LmCloudSpmdPipeline):
  """SPMD-pipelined model with 9B params.

  Global batch size = 4 * 16 * 8 = 512
  """
  MICROBATCH_SIZE = 4
  PERCORE_BATCH_SIZE = 8

  NUM_STAGES = 16
  NUM_LAYERS = 48
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  # 16-way pipeline and 4-way data parallelism.
  MESH_SHAPE = [16, 4, 1, 1]
