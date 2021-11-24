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
"""Decoder-only language model configurations."""

import math
from typing import List

import jax
import jax.numpy as jnp
from lingvo.jax import base_input
from lingvo.jax import base_model_params
from lingvo.jax import layers
from lingvo.jax import model
from lingvo.jax import model_registry
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import schedules
from lingvo.jax.tasks.lm import input_generator
import numpy as np

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

  def datasets(self) -> List[InstantiableParams]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]


## Data parallel training.


@model_registry.register_model
class LmCloudTransformerAdam(SyntheticDataset):
  r"""32-layer Transformer LM using Adam."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  ENABLE_WHILE_LOOP = True

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    vocab_size = self.VOCAB_SIZE
    num_layers = self.NUM_LAYERS
    num_heads = self.NUM_HEADS
    model_dims = self.MODEL_DIMS
    hidden_dims = self.HIDDEN_DIMS
    dropout_prob = self.DROPOUT_PROB

    model_p = model.LanguageModel.Params().Set(name='xformer_lm')
    model_p.lm.packed_input = True
    model_p.lm.model_dims = model_dims
    model_p.lm.hidden_dims = hidden_dims
    model_p.lm.num_layers = num_layers
    model_p.lm.num_heads = num_heads
    model_p.lm.vocab_size = vocab_size
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.dropout_prob = dropout_prob
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(model_dims))
    model_p.lm.softmax_tpl.params_init = softmax_init

    lp = model_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.Adam.Params().Set(
        beta1=0.9,
        beta2=0.99,
        weight_decay=1e-3,
        clip_gradient_norm_to_value=5.0)
    lp.optimizer.learning_rate = self.LEARNING_RATE
    lp.optimizer.lr_schedule = (
        schedules.LinearRampupExponentialDecay.Params().Set(
            warmup=4000,
            decay_start=4001,
            decay_end=300000,
            min_ratio=0.1,
            max=1.0))
    return model_p


@model_registry.register_model
class LmCloudTransformerAdamTest(LmCloudTransformerAdam):
  NUM_LAYERS = 2


## SPMD Model parallel training.


class LmCloudSpmd(SyntheticDataset):
  r"""Base config for an SPMD model."""

  NUM_LAYERS = 10
  MODEL_DIMS = 2048

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    vocab_size = 32000
    num_layers = self.NUM_LAYERS
    model_dims = self.MODEL_DIMS
    hidden_dims = model_dims * 4
    dims_per_head = 128
    assert model_dims % dims_per_head == 0
    num_heads = int(model_dims / dims_per_head)
    dropout_prob = 0.0

    model_p = model.LanguageModel.Params().Set(name='xformer_lm')
    model_p.lm.packed_input = True
    model_p.lm.model_dims = model_dims
    model_p.lm.hidden_dims = hidden_dims
    model_p.lm.num_layers = num_layers
    model_p.lm.num_heads = num_heads
    model_p.lm.vocab_size = vocab_size
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0

    model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    model_p.lm.stacked_transformer_tpl.enable_while_loop = True
    model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
        self.CHECKPOINT_POLICY)

    model_p.lm.stacked_transformer_tpl.dropout_prob = dropout_prob
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_fflayer_tpl.activation = 'GELU'
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(model_dims))
    model_p.lm.softmax_tpl.params_init = softmax_init

    # Enable bf16.
    model_p.fprop_dtype = jnp.bfloat16

    lp = model_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.Adam.Params().Set(
        beta1=0.9,
        beta2=0.99,
        weight_decay=1e-3,
        clip_gradient_norm_to_value=5.0)
    lp.optimizer.learning_rate = 2.5e-4
    lp.optimizer.lr_schedule = (
        schedules.LinearRampupExponentialDecay.Params().Set(
            warmup=4000,
            decay_start=4001,
            decay_end=300000,
            min_ratio=0.1,
            max=1.0))

    # Set sharding annotations.
    mesh_shape = self.MESH_SHAPE
    device_count = np.prod(mesh_shape)
    device_ids_mesh = np.arange(device_count).reshape(mesh_shape)
    model_p.device_mesh = device_ids_mesh
    replica_axis = 'replica'
    data_axis = 'data'
    mdl_axis = 'mdl'
    mesh_axis_names = [replica_axis, data_axis, mdl_axis]
    model_p.train.inputs_split_mapping = NestedMap(
        map_1d=((replica_axis, data_axis),),
        map_2d=((replica_axis, data_axis), None))
    model_p.train.decoder_inputs_split_mapping = NestedMap(
        map_1d=((replica_axis, data_axis),))
    model_p.train.decoder_states_split_mapping = NestedMap(
        map_0d=None,
        map_4d=(None, (replica_axis, data_axis), mdl_axis, None),
        # 5d inputs are for the decoder states of shape [layers, seq_len,
        # batch_size, num_heads, dims_per_head]
        map_5d=(None, None, (replica_axis, data_axis), mdl_axis, None),
    )
    model_p.train.save_interval_steps = 5000
    model_p.mesh_axis_names = mesh_axis_names
    model_p.lm = model_p.lm.cls.set_sharding_params_v1(
        model_p.lm,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        device_ids_mesh=device_ids_mesh,
        mesh_axis_names=mesh_axis_names)

    return model_p


@model_registry.register_model
class LmCloudSpmdTest(LmCloudSpmd):
  r"""SPMD model with small params for local CPU test run.

  Global batch size = 1 * 1 * 1 * 4 = 4
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 2
  MODEL_DIMS = 64

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 1, 1]


@model_registry.register_model
class LmCloudSpmd2B(LmCloudSpmd):
  r"""SPMD model with 2B params.

  Global batch size = 2 * 2 * 1 * 32 = 128
  """
  PERCORE_BATCH_SIZE = 32

  NUM_LAYERS = 18
  MODEL_DIMS = 3072

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 4, 1]


@model_registry.register_model
class LmCloudSpmd32B(LmCloudSpmd):
  r"""SPMD model with 32B params.

  Global batch size = 4 * 4 * 4 * 8 = 512
  """
  PERCORE_BATCH_SIZE = 8

  NUM_LAYERS = 40
  MODEL_DIMS = 8192

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 16, 4]


@model_registry.register_model
class LmCloudSpmd64B(LmCloudSpmd):
  r"""SPMD model with 64B params.

  Global batch size = 4 * 4 * 8 * 8 = 1024
  """
  PERCORE_BATCH_SIZE = 8

  NUM_LAYERS = 51
  MODEL_DIMS = 10240

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 16, 8]


@model_registry.register_model
class LmCloudSpmd128B(LmCloudSpmd):
  r"""SPMD model with 128B params.

  Global batch size = 4 * 8 * 8 * 4 = 1024
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 71
  MODEL_DIMS = 12288

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 64, 4]


@model_registry.register_model
class LmCloudSpmd256B(LmCloudSpmd):
  r"""SPMD model with 256B params.

  Global batch size = 4 * 8 * 8 * 8 = 2048
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 80
  MODEL_DIMS = 16384

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 64, 8]


@model_registry.register_model
class LmCloudSpmd512B(LmCloudSpmd):
  r"""SPMD model with 512B params.

  Global batch size = 4 * 8 * 8 * 16 = 4096
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 102
  MODEL_DIMS = 20480

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 64, 16]


@model_registry.register_model
class LmCloudSpmd1024B(LmCloudSpmd):
  r"""SPMD model with 1024B params.

  Global batch size = 2 * 8 * 16 * 16 = 4096
  """
  PERCORE_BATCH_SIZE = 2

  NUM_LAYERS = 142
  MODEL_DIMS = 24576

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  MESH_SHAPE = [1, 256, 8]
