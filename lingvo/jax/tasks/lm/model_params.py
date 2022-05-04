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
from lingvo.jax import base_model
from lingvo.jax import base_model_params
from lingvo.jax import base_task
from lingvo.jax import layers
from lingvo.jax import optimizers
from lingvo.jax import py_utils
from lingvo.jax import schedules
import numpy as np

InstantiableParams = py_utils.InstantiableParams
NestedMap = py_utils.NestedMap
WeightInit = py_utils.WeightInit


def set_sharding_annotations_v1(task_p: InstantiableParams,
                                mesh_shape: Sequence[int]) -> None:
  """Sets the sharding annotations in the task config for the given mesh.

  Args:
    task_p: The task parameters to update with sharding annotations.
    mesh_shape: a 3D sequence representing the mesh shape.
  """
  model_p = task_p.model
  asserts.eq(len(mesh_shape), 3)
  device_count = np.prod(mesh_shape)
  device_ids_mesh = np.arange(device_count).reshape(mesh_shape)
  model_p.device_mesh = device_ids_mesh
  replica_axis = 'replica'
  data_axis = 'data'
  mdl_axis = 'mdl'
  mesh_axis_names = [replica_axis, data_axis, mdl_axis]
  task_p.train.inputs_split_mapping = NestedMap(
      map_1d=((replica_axis, data_axis),),
      map_2d=((replica_axis, data_axis), None))
  model_p.mesh_axis_names = mesh_axis_names
  if hasattr(model_p, 'lm'):
    model_p.lm = model_p.lm.cls.set_sharding_params_v1(
        model_p.lm,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        device_ids_mesh=device_ids_mesh,
        mesh_axis_names=mesh_axis_names)


def set_default_adam(task_p: InstantiableParams, learning_rate: float,
                     weight_decay: float) -> None:
  """Sets the default Adam optimizer settings in the model config.

  Args:
    task_p: The task parameters to update with optimizer specs.
    learning_rate: The learning rate to set.
    weight_decay: The weight_decay to set.
  """
  lp = task_p.train.learner
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


def set_default_adafactor(task_p: InstantiableParams,
                          learning_rate: float,
                          weight_decay: float,
                          clip_gradient_norm_to_value: float = 5.0) -> None:
  """Sets the default AdaFactor optimizer settings in the task config.

  Args:
    task_p: The task parameters to update with optimizer specs.
    learning_rate: The learning rate to set.
    weight_decay: The weight_decay to set.
    clip_gradient_norm_to_value: clip_gradient_norm_to_value.
  """
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = optimizers.ShardedAdafactor.Params().Set(
      decay_method='adam',
      beta1=0.9,
      decay_adam=0.99,
      weight_decay=weight_decay,
      clip_gradient_norm_to_value=clip_gradient_norm_to_value)
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


class ClassificationModelAdam(base_model_params.BaseModelParams):
  """A simple MLP language model configuration using Adam."""
  NUM_LAYER = 8
  INPUT_DIM = 4096
  HIDDEN_DIM = 7168
  OUTPUT_DIM = 4096
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  CHECKPOINT_EVERY_N_STEPS = 5
  SUMMARY_INTERVAL_STEPS = 5
  NUM_TRAIN_STEPS = 10
  MLP_WEIGHT_SHARDING = None
  SOFTMAX_WEIGHT_SHARDING = None

  # sub-class specify a mesh to use SPMD
  MESH_SHAPE = None
  NUM_DEVICES = None

  def task(self) -> InstantiableParams:
    task_p = base_task.SingleTask.Params().Set(name='classification_task')
    task_p.model = base_model.ClassificationMLPModel.Params().Set(
        name='classification_model')
    model_p = task_p.model
    model_p.mlp_tpl.ff_tpl.input_dims = self.INPUT_DIM
    model_p.mlp_tpl.ff_tpl.output_dims = self.OUTPUT_DIM
    model_p.mlp_tpl.hidden_dims = self.HIDDEN_DIM
    model_p.mlp_tpl.num_layers = self.NUM_LAYER
    model_p.softmax_tpl.input_dims = self.INPUT_DIM
    model_p.softmax_tpl.num_classes = self.INPUT_DIM
    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    model_p.device_mesh = np.arange(self.NUM_DEVICES).reshape(self.MESH_SHAPE)
    model_p.mesh_axis_names = ['x', 'y', 'z']
    model_p.softmax_tpl.weight_split_dims_mapping.wt = self.SOFTMAX_WEIGHT_SHARDING
    model_p.mlp_tpl.device_mesh = model_p.device_mesh
    model_p.mlp_tpl.weight_split_dims_mapping.wt = self.MLP_WEIGHT_SHARDING
    set_sharding_annotations_v1(task_p, self.MESH_SHAPE)
    set_default_adam(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)
    task_p.train.num_train_steps = self.NUM_TRAIN_STEPS
    return task_p


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
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ACTIVATION_FUNCTION = 'RELU'
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000

  ENABLE_BFLOAT16 = True

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    task_p = base_task.SingleTask.Params().Set(name='bert_task')
    task_p.model = base_model.BertModel.Params().Set(name='bert_lm')
    model_p = task_p.model
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
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    if self.USE_REPEATED_LAYER:
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

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)

    set_default_adam(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    return task_p


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
    task_p = base_task.SingleTask.Params().Set(name='bert_task')
    task_p.model = base_model.BertModel.Params().Set(name='bert_lm')
    model_p = task_p.model
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
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    if self.USE_REPEATED_LAYER:
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

    task_p.train.save_max_to_keep = self.CHECKPOINT_SAVE_MAX_TO_KEEP

    set_default_adafactor(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)
    set_sharding_annotations_v1(task_p, self.MESH_SHAPE)

    return task_p


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
  USE_REPEATED_LAYER = False
  ACTIVATION_FUNCTION = 'RELU'

  PACKED_INPUT = True
  ATTEN_LOGIT_CAP = 50.0
  USE_BIAS = False

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    task_p = base_task.SingleTask.Params().Set(name='xformer_task')
    task_p.model = base_model.LanguageModel.Params().Set(name='xformer_lm')
    model_p = task_p.model
    model_p.lm.packed_input = self.PACKED_INPUT
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.vocab_size = self.VOCAB_SIZE
    model_p.lm.softmax_tpl.scale_sqrt_depth = True

    stacked_transformer_tpl = layers.StackedTransformer.Params()
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
      stacked_transformer_tpl.num_layers = 1
      model_p.lm.stacked_transformer_tpl.block = (stacked_transformer_tpl)
      model_p.lm.stacked_transformer_tpl.x_times = self.NUM_LAYERS
    else:
      model_p.lm.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init

    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)
    set_default_adam(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)

    return task_p


class TransformerLmSpmdAdafactor(base_model_params.BaseModelParams):
  """Base SPMD Transformer LM configuration using Adafactor."""
  # architecture related
  NUM_LAYERS = 10
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = None
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16

  USE_REPEATED_LAYER = False
  TRAINABLE_POSITION_EMB = False
  TRAINABLE_PE_MAX_SEQ_LEN = 16 * 1024
  RELATIVE_BIAS = False
  USE_ROTARY_POSITION_EMB = False
  NORM_POLICY = 'pre'
  ENABLE_DCONV = False
  COMBINE_QKV = True
  ACTIVATION = 'RELU'

  # optimizer related
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  CLIP_GRADIENT_NORM_TO_VALUE = 5.0
  WEIGHT_DECAY = 1e-3
  SOFTMAX_CAP_LOGITS = 30.0
  ATTEN_LOGIT_CAP = 50.0
  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # checkpoint
  CHECKPOINT_EVERY_N_STEPS = 5000
  SUMMARY_INTERVAL_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    if self.DIMS_PER_HEAD is not None:
      if self.NUM_HEADS is None:
        assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
        num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
      else:
        assert self.MODEL_DIMS == self.NUM_HEADS * self.DIMS_PER_HEAD
    else:
      assert self.NUM_HEADS is not None
      num_heads = self.NUM_HEADS

    task_p = base_task.SingleTask.Params().Set(name='xformer_task')
    task_p.model = base_model.LanguageModel.Params().Set(name='xformer_lm')
    model_p = task_p.model
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.vocab_size = self.VOCAB_SIZE

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    if self.TRAINABLE_POSITION_EMB:
      model_p.lm.position_emb_tpl = (
          layers.TrainablePositionalEmbedding.Params().Set(
              max_seq_length=self.TRAINABLE_PE_MAX_SEQ_LEN))

    stacked_transformer_tpl = layers.StackedTransformer.Params()
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION
    transformer_layer_p.tr_atten_tpl.dconv_qkv = self.ENABLE_DCONV

    # Only one of RELATIVE_BIAS or USE_ROTARY_POSITION_EMB can be True.
    assert (not self.RELATIVE_BIAS) or (not self.USE_ROTARY_POSITION_EMB)
    if self.RELATIVE_BIAS:
      transformer_layer_p.tr_atten_tpl.relative_bias_tpl = (
          layers.RelativeBias.Params())
    if self.USE_ROTARY_POSITION_EMB:
      transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.Params())
      stacked_transformer_tpl.num_layers = 1
      model_p.lm.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm.stacked_transformer_tpl = stacked_transformer_tpl

    # Enable bf16.
    model_p.fprop_dtype = self.FPROP_DTYPE

    set_default_adafactor(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY,
                          self.CLIP_GRADIENT_NORM_TO_VALUE)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.save_max_to_keep = self.CHECKPOINT_MAX_TO_KEEP

    if self.MESH_SHAPE is not None:
      set_sharding_annotations_v1(task_p, self.MESH_SHAPE)
    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)

    return task_p


class TransformerLmSpmdPipelineAdafactor(TransformerLmSpmdAdafactor):
  """Base SPMD pipelined Transformer LM configuration using Adafactor."""
  # architecture related
  NUM_LAYERS = 10
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = None
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16

  # Default these flags to False as we already have a loop over stages.
  USE_REPEATED_LAYER = False
  TRAINABLE_POSITION_EMB = False
  TRAINABLE_PE_MAX_SEQ_LEN = 16 * 1024
  RELATIVE_BIAS = False
  USE_ROTARY_POSITION_EMB = False
  NORM_POLICY = 'pre'
  ENABLE_DCONV = False
  COMBINE_QKV = True
  ACTIVATION = 'RELU'

  # optimizer related
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  CLIP_GRADIENT_NORM_TO_VALUE = 5.0
  WEIGHT_DECAY = 1e-3
  SOFTMAX_CAP_LOGITS = 30.0
  ATTEN_LOGIT_CAP = 50.0
  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # checkpoint
  CHECKPOINT_EVERY_N_STEPS = 5000
  SUMMARY_INTERVAL_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  # Pipeline related.
  NUM_STAGES = None
  # One of the two need to be set.
  NUM_MICROBATCHES = None
  MICROBATCH_SIZE = None

  # Sub-class has to specify a mesh with shape [NUM_STAGES, replica, data, mdl]
  MESH_SHAPE = None

  def task(self) -> InstantiableParams:
    """Returns the task parameters."""
    if self.DIMS_PER_HEAD is not None:
      if self.NUM_HEADS is None:
        assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
        num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
      else:
        assert self.MODEL_DIMS == self.NUM_HEADS * self.DIMS_PER_HEAD
    else:
      assert self.NUM_HEADS is not None
      num_heads = self.NUM_HEADS

    assert self.NUM_STAGES is not None
    assert self.NUM_LAYERS % self.NUM_STAGES == 0
    assert self.NUM_MICROBATCHES is not None or self.MICROBATCH_SIZE is not None
    assert self.MESH_SHAPE is not None and len(self.MESH_SHAPE) == 4
    assert self.MESH_SHAPE[0] == self.NUM_STAGES

    task_p = base_task.SingleTask.Params().Set(name='xformer_task')
    task_p.model = base_model.LanguageModel.Params().Set(name='xformer_lm')
    model_p = task_p.model
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.vocab_size = self.VOCAB_SIZE

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm.softmax_tpl.params_init = softmax_init
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    if self.TRAINABLE_POSITION_EMB:
      model_p.lm.position_emb_tpl = (
          layers.TrainablePositionalEmbedding.Params().Set(
              max_seq_length=self.TRAINABLE_PE_MAX_SEQ_LEN))

    stacked_transformer_tpl = layers.StackedTransformer.Params()
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS // self.NUM_STAGES
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION
    transformer_layer_p.tr_atten_tpl.dconv_qkv = self.ENABLE_DCONV

    # Only one of RELATIVE_BIAS or USE_ROTARY_POSITION_EMB can be True.
    assert (not self.RELATIVE_BIAS) or (not self.USE_ROTARY_POSITION_EMB)
    if self.RELATIVE_BIAS:
      transformer_layer_p.tr_atten_tpl.relative_bias_tpl = (
          layers.RelativeBias.Params())
    if self.USE_ROTARY_POSITION_EMB:
      transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    if self.USE_REPEATED_LAYER:
      stacked_transformer_tpl = layers.StackedTransformerRepeated.Params().Set(
          block=stacked_transformer_tpl.Set(num_layers=1))
      stacked_transformer_tpl.x_times = self.NUM_LAYERS // self.NUM_STAGES
      stacked_transformer_tpl.checkpoint_policy = self.CHECKPOINT_POLICY

    # Wrap it with a pipeline layer.
    model_p.lm.stacked_transformer_tpl = layers.PipelinedTransformer.Params(
    ).Set(
        pipeline_stage=stacked_transformer_tpl,
        num_pipeline_stages=self.NUM_STAGES,
        num_pipeline_microbatches=self.NUM_MICROBATCHES,
        pipeline_microbatch_size=self.MICROBATCH_SIZE)

    # Enable bf16.
    model_p.fprop_dtype = self.FPROP_DTYPE

    set_default_adafactor(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY,
                          self.CLIP_GRADIENT_NORM_TO_VALUE)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.save_max_to_keep = self.CHECKPOINT_MAX_TO_KEEP
    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl.pipeline_stage)

    # Set up the sharding specifications.
    device_count = np.prod(self.MESH_SHAPE)
    device_ids_mesh = np.arange(device_count).reshape(self.MESH_SHAPE)
    model_p.device_mesh = device_ids_mesh
    stage_axis = 'stage'
    replica_axis = 'replica'
    data_axis = 'data'
    mdl_axis = 'mdl'
    mesh_axis_names = [stage_axis, replica_axis, data_axis, mdl_axis]
    model_p.mesh_axis_names = mesh_axis_names

    # Set in-stage layer shardings.
    model_p.lm = model_p.lm.cls.set_sharding_params_v1(
        model_p.lm,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        device_ids_mesh=device_ids_mesh,
        mesh_axis_names=mesh_axis_names)

    # Include stage_axis in input partitioning to allow full data parallelism in
    # embedding layers.
    task_p.train.inputs_split_mapping = NestedMap(
        map_1d=((stage_axis, replica_axis, data_axis),),
        map_2d=((stage_axis, replica_axis, data_axis), None))

    # Run softmax/embedding in data parallelism across all cores.
    softmax_p = model_p.lm.softmax_tpl
    softmax_p.activation_split_dims_mapping.emb_out_split_dims_mapping = [
        (stage_axis, replica_axis, data_axis), None, mdl_axis
    ]
    softmax_p.activation_split_dims_mapping.out = [(stage_axis, replica_axis,
                                                    data_axis), None, mdl_axis]

    pipeline_layer_p = model_p.lm.stacked_transformer_tpl
    pipeline_layer_p.weight_split_dims_mapping.stages = [stage_axis]
    # Match the final output sharding to softmax input sharding.
    pipeline_layer_p.activation_split_dims_mapping.final_out = [
        (stage_axis, replica_axis, data_axis), None, mdl_axis
    ]

    return task_p
