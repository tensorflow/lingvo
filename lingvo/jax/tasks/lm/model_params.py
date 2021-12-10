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
"""Base language model configurations."""

import math
from typing import Sequence

from jax import numpy as jnp
from lingvo.jax import asserts
from lingvo.jax import base_model_params
from lingvo.jax import layers
from lingvo.jax import model
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import schedules
import numpy as np

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit


def set_sharding_annotations_v1(model_p: InstantiableParams,
                                mesh_shape: Sequence[int]) -> None:
  """Sets the sharding annotations in the model config for the given mesh.

  Args:
    model_p: The model parameters to update with sharding annotations.
    mesh_shape: a 3D sequence representing the mesh shape.
  """
  asserts.eq(len(mesh_shape), 3)
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
  model_p.mesh_axis_names = mesh_axis_names
  model_p.lm = model_p.lm.cls.set_sharding_params_v1(
      model_p.lm,
      replica_axis=replica_axis,
      data_axis=data_axis,
      mdl_axis=mdl_axis,
      device_ids_mesh=device_ids_mesh,
      mesh_axis_names=mesh_axis_names)


def set_default_adam(model_p: InstantiableParams, learning_rate: float,
                     weight_decay: float) -> None:
  """Sets the default Adam optimizer settings in the model config.

  Args:
    model_p: The model parameters to update with optimizer specs.
    learning_rate: The learning rate to set.
    weight_decay: The weight_decay to set.
  """
  lp = model_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = optimizers.Adam.Params().Set(
      beta1=0.9,
      beta2=0.99,
      weight_decay=weight_decay,
      clip_gradient_norm_to_value=5.0)
  lp.optimizer.learning_rate = learning_rate
  lp.optimizer.lr_schedule = (
      schedules.LinearRampupExponentialDecay.Params().Set(
          warmup=4000,
          decay_start=4001,
          decay_end=300000,
          min_ratio=0.1,
          max=1.0))


def set_default_adafactor(model_p: InstantiableParams, learning_rate: float,
                          weight_decay: float) -> None:
  """Sets the default AdaFactor optimizer settings in the model config.

  Args:
    model_p: The model parameters to update with optimizer specs.
    learning_rate: The learning rate to set.
    weight_decay: The weight_decay to set.
  """
  lp = model_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = optimizers.ShardedAdafactor.Params().Set(
      decay_method='adam',
      beta1=0.9,
      decay_adam=0.99,
      weight_decay=weight_decay,
      clip_gradient_norm_to_value=5.0)
  lp.optimizer.learning_rate = learning_rate
  lp.optimizer.lr_schedule = (
      schedules.LinearRampupExponentialDecay.Params().Set(
          warmup=4000,
          decay_start=4001,
          decay_end=100000,
          min_ratio=0.1,
          max=1.0))


class TransformerBertPmapAdam(base_model_params.BaseModelParams):
  """Base Pmap Transformer Bert configuration using Adam."""

  NUM_LAYERS = 4
  VOCAB_SIZE = 32000
  NUM_HEADS = 8
  MODEL_DIMS = 128
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  ENABLE_WHILE_LOOP = True
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ACTIVATION_FUNCTION = 'RELU'
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000

  ENABLE_BFLOAT16 = True

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = model.BertModel.Params().Set(name='bert_lm')
    model_p.lm.masked_lm = True
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.hidden_dims = self.HIDDEN_DIMS
    model_p.lm.num_layers = self.NUM_LAYERS
    model_p.lm.num_heads = self.NUM_HEADS
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0
    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
    else:
      model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
        self.CHECKPOINT_POLICY)
    model_p.lm.stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    set_default_adam(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    return model_p


class TransformerBertSpmdAdafactor(base_model_params.BaseModelParams):
  """Base SPMD Transformer Bert configuration using AdaFactor."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  ENABLE_WHILE_LOOP = True
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ENABLE_BFLOAT16 = False
  MASK_TOKEN_ID = 0

  ACTIVATION_FUNCTION = 'RELU'

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 500
  CHECKPOINT_SAVE_MAX_TO_KEEP = 10

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = model.BertModel.Params().Set(name='bert_lm')
    model_p.mask_token_id = self.MASK_TOKEN_ID
    model_p.lm.masked_lm = True
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.hidden_dims = self.HIDDEN_DIMS
    model_p.lm.num_layers = self.NUM_LAYERS
    model_p.lm.num_heads = self.NUM_HEADS
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0
    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
    else:
      model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
        self.CHECKPOINT_POLICY)
    model_p.lm.stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    model_p.train.save_max_to_keep = self.CHECKPOINT_SAVE_MAX_TO_KEEP

    set_default_adafactor(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    set_sharding_annotations_v1(model_p, self.MESH_SHAPE)

    return model_p


class TransformerLmPmapAdam(base_model_params.BaseModelParams):
  """Base Pmap Transformer LM configuration using Adam."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  ENABLE_WHILE_LOOP = True

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = model.LanguageModel.Params().Set(name='xformer_lm')
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.hidden_dims = self.HIDDEN_DIMS
    model_p.lm.num_layers = self.NUM_LAYERS
    model_p.lm.num_heads = self.NUM_HEADS
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    set_default_adam(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    return model_p


class TransformerLmSpmdAdam(base_model_params.BaseModelParams):
  """Base SPMD Transformer LM configuration using Adam."""

  NUM_LAYERS = 10
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  WEIGHT_DECAY = 1e-3
  ENABLE_WHILE_LOOP = True

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
    num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
    dropout_prob = self.DROPOUT_PROB

    model_p = model.LanguageModel.Params().Set(name='xformer_lm')
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.hidden_dims = self.HIDDEN_DIMS
    model_p.lm.num_layers = self.NUM_LAYERS
    model_p.lm.num_heads = num_heads
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0

    model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
        self.CHECKPOINT_POLICY)

    model_p.lm.stacked_transformer_tpl.dropout_prob = dropout_prob
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_fflayer_tpl.activation = 'GELU'
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    # Enable bf16.
    model_p.fprop_dtype = jnp.bfloat16

    set_default_adam(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    model_p.train.save_interval_steps = 5000

    set_sharding_annotations_v1(model_p, self.MESH_SHAPE)

    return model_p
