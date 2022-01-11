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
from lingvo.jax import base_layer
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


def maybe_setup_moe_params(model_p: InstantiableParams):
  """Convert a FeedforwardLayer to a MoE Layer for StackedTransformer."""
  if model_p.cls == layers.StackedTransformerRepeated:
    model_p = model_p.block

  if model_p.num_experts == 0:
    return model_p

  ff_p = model_p.transformer_layer_params_tpl.tr_fflayer_tpl
  assert issubclass(ff_p.cls, layers.TransformerFeedForward)
  moe_p = model_p.moe_layer_tpl
  # Copy over the base params.
  base_layer.BaseLayer.copy_base_params(ff_p, moe_p)
  # Copy over othe params.
  moe_p.name = ff_p.name
  moe_p.input_dims = ff_p.input_dims
  moe_p.hidden_dims = ff_p.hidden_dims
  moe_p.ln_tpl = ff_p.ln_tpl.Copy()
  moe_p.activation = ff_p.activation
  moe_p.relu_dropout_tpl = ff_p.relu_dropout_tpl.Copy()
  moe_p.relu_dropout_prob = ff_p.relu_dropout_prob
  moe_p.residual_dropout_tpl = ff_p.residual_dropout_tpl.Copy()
  moe_p.residual_dropout_prob = ff_p.residual_dropout_prob
  moe_p.add_skip_connection = ff_p.add_skip_connection
  moe_p.norm_policy = ff_p.norm_policy


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
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0

    stacked_transformer_tpl = layers.StackedTransformer.Params()

    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.enable_while_loop = (self.ENABLE_WHILE_LOOP)
    stacked_transformer_tpl.checkpoint_policy = (self.CHECKPOINT_POLICY)
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    if self.USE_REPEATED_LAYER:
      assert not self.ENABLE_WHILE_LOOP
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
      stacked_transformer_tpl.num_layers = 1
      model_p.lm.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)

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
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0

    stacked_transformer_tpl = layers.StackedTransformer.Params()
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.enable_while_loop = (self.ENABLE_WHILE_LOOP)
    stacked_transformer_tpl.checkpoint_policy = (self.CHECKPOINT_POLICY)
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    if self.USE_REPEATED_LAYER:
      assert not self.ENABLE_WHILE_LOOP
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
      stacked_transformer_tpl.num_layers = 1
      model_p.lm.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    model_p.train.save_max_to_keep = self.CHECKPOINT_SAVE_MAX_TO_KEEP

    set_default_adafactor(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)
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
  ACTIVATION_FUNCTION = 'RELU'

  PACKED_INPUT = True
  ATTEN_LOGIT_CAP = 50.0
  USE_BIAS = False

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = model.LanguageModel.Params().Set(name='xformer_lm')
    model_p.lm.packed_input = self.PACKED_INPUT
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.stacked_transformer_tpl = layers.StackedTransformer.Params()
    stacked_transformer_tpl = model_p.lm.stacked_transformer_tpl
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS

    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)
    set_default_adam(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    return model_p


class TransformerLmSpmdAdafactor(base_model_params.BaseModelParams):
  """Base SPMD Transformer LM configuration using Adam."""

  NUM_LAYERS = 10
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = None
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  ATTEN_LOGIT_CAP = 50.0
  LEARNING_RATE = 2.5e-4
  WEIGHT_DECAY = 1e-3
  ENABLE_WHILE_LOOP = True
  USE_REPEATED_LAYER = False
  SOFTMAX_CAP_LOGITS = 30.0
  ATTEN_LOGIT_CAP = 50.0
  FPROP_DTYPE = jnp.bfloat16
  COMBINE_QKV = True
  ACTIVATION = 'RELU'

  CHECKPOINT_EVERY_N_STEPS = 5000

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    if self.DIMS_PER_HEAD is not None:
      assert self.NUM_HEADS is None
      assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
      num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
    else:
      assert self.NUM_HEADS is not None
      assert self.DIMS_PER_HEAD is None
      num_heads = self.NUM_HEADS

    dropout_prob = self.DROPOUT_PROB

    model_p = model.LanguageModel.Params().Set(name='xformer_lm')
    model_p.lm.packed_input = True
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    stacked_transformer_tpl = layers.StackedTransformer.Params()
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = num_heads

    stacked_transformer_tpl.enable_while_loop = (self.ENABLE_WHILE_LOOP)
    stacked_transformer_tpl.checkpoint_policy = (self.CHECKPOINT_POLICY)

    stacked_transformer_tpl.dropout_prob = dropout_prob
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION

    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
      stacked_transformer_tpl.num_layers = 1
      model_p.lm.stacked_transformer_tpl.block = (stacked_transformer_tpl)
      model_p.lm.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    # Enable bf16.
    model_p.fprop_dtype = self.FPROP_DTYPE

    set_default_adafactor(model_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    set_sharding_annotations_v1(model_p, self.MESH_SHAPE)
    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)

    return model_p
