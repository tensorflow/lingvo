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
"""BERT masked language model configurations."""

import math
from typing import List

import jax
from jax import numpy as jnp
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


class BertDataset(base_model_params.BaseModelParams):
  """MLPerf Bert dataset."""
  PERCORE_BATCH_SIZE = 8
  MLPERF_REMASK = True
  RANDOM_BUFFER_SIZE = 100_000

  def _DatasetTrain(self) -> InstantiableParams:
    """Parameters for using the original ML Perf training data."""
    p = input_generator.TFRecordBertInput.Params()
    p.name = 'train'
    p.input_file = 'gs://mlperf_v1_1/bert/train'
    p.read_as_eval_data = False
    p.enable_packing = True
    num_local_devices = jax.local_device_count()
    p.batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    p.remask = self.MLPERF_REMASK
    p.mlm_augmenter.Set(mask_token_id=103, vocab_size=30522)
    p.num_samples = 156_725_653
    p.file_buffer_size = self.RANDOM_BUFFER_SIZE
    return p

  def _DatasetTest(self) -> InstantiableParams:
    """Parameters for using the original ML Perf eval data."""
    p = input_generator.TFRecordBertInput.Params()
    p.name = 'test'
    p.input_file = 'gs://mlperf_v1_1/bert/eval'
    p.read_as_eval_data = True
    p.enable_packing = False
    p.resettable = True
    num_local_devices = jax.local_device_count()
    p.batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    p.num_samples = 10_000
    return p

  def Datasets(self) -> List[base_model_params.DatasetParams]:
    """Returns a list of dataset parameters."""
    ds_params_train = base_model_params.DatasetParams().Set(
        name='train', is_training=True, input_gen_params=self._DatasetTrain())
    ds_params_test = base_model_params.DatasetParams().Set(
        name='test', is_training=False, input_gen_params=self._DatasetTest())
    return [ds_params_train, ds_params_test]

  def Task(self) -> InstantiableParams:
    raise NotImplementedError()


@model_registry.RegisterModel
class BertAdamL4H128(BertDataset):
  r"""4-layer Transformer LM using Adam on JF 2x2.

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
  ENABLE_WHILE_LOOP = True
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ACTIVATION_FUNCTION = 'RELU'
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000

  ENABLE_BFLOAT16 = True

  def Task(self) -> InstantiableParams:
    """Returns the task parameters."""
    vocab_size = self.VOCAB_SIZE
    num_layers = self.NUM_LAYERS
    num_heads = self.NUM_HEADS
    model_dims = self.MODEL_DIMS
    hidden_dims = self.HIDDEN_DIMS
    dropout_prob = self.DROPOUT_PROB

    model_p = model.BertModel.Params().Set(name='bert_lm')
    model_p.lm.masked_lm = True
    model_p.lm.packed_input = True
    model_p.lm.model_dims = model_dims
    model_p.lm.hidden_dims = hidden_dims
    model_p.lm.num_layers = num_layers
    model_p.lm.num_heads = num_heads
    model_p.lm.vocab_size = vocab_size
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0
    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerLayersRepeated.Params())
    else:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerLayers.Params())
    model_p.lm.stacked_transformer_tpl.enable_while_loop = (
        self.ENABLE_WHILE_LOOP)
    model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
        self.CHECKPOINT_POLICY)
    model_p.lm.stacked_transformer_tpl.dropout_prob = dropout_prob
    transformer_layer_p = (
        model_p.lm.stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(model_dims))
    model_p.lm.softmax_tpl.params_init = softmax_init

    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    lp = model_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.AdamOptimizer.Params().Set(
        beta1=0.9,
        beta2=0.99,
        weight_decay=self.WEIGHT_DECAY,
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


class BertSpmd(BertDataset):
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

  def Task(self) -> InstantiableParams:
    """Returns the task parameters."""
    vocab_size = self.VOCAB_SIZE
    num_layers = self.NUM_LAYERS
    num_heads = self.NUM_HEADS
    model_dims = self.MODEL_DIMS
    hidden_dims = self.HIDDEN_DIMS
    dropout_prob = self.DROPOUT_PROB

    model_p = model.BertModel.Params().Set(name='bert_lm')
    model_p.mask_token_id = self.MASK_TOKEN_ID
    model_p.lm.masked_lm = True
    model_p.lm.packed_input = True
    model_p.lm.model_dims = model_dims
    model_p.lm.hidden_dims = hidden_dims
    model_p.lm.num_layers = num_layers
    model_p.lm.num_heads = num_heads
    model_p.lm.vocab_size = vocab_size
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = 30.0
    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerLayersRepeated.Params())
    else:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerLayers.Params())
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
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION_FUNCTION
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(model_dims))
    model_p.lm.softmax_tpl.params_init = softmax_init

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    model_p.train.save_max_to_keep = self.CHECKPOINT_SAVE_MAX_TO_KEEP
    lp = model_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.ShardedAdafactorOptimizer.Params().Set(
        decay_method='adam',
        beta1=0.9,
        decay_adam=0.99,
        weight_decay=self.WEIGHT_DECAY,
        clip_gradient_norm_to_value=5.0)
    lp.optimizer.learning_rate = self.LEARNING_RATE
    lp.optimizer.lr_schedule = (
        schedules.LinearRampupExponentialDecay.Params().Set(
            warmup=4000,
            decay_start=4001,
            decay_end=100000,
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
    model_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    model_p.mesh_axis_names = mesh_axis_names
    model_p.lm = model_p.lm.cls.SetShardingParamsV1(
        model_p.lm,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        device_ids_mesh=device_ids_mesh,
        mesh_axis_names=mesh_axis_names)
    return model_p


@model_registry.RegisterModel
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
  ENABLE_WHILE_LOOP = True
  ENABLE_BFLOAT16 = True

  # Sub-class has to specify a mesh.
  MESH_SHAPE = [1, 4, 2]

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000


@model_registry.RegisterModel
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
  ENABLE_WHILE_LOOP = True
  ENABLE_BFLOAT16 = True
  USE_MLPERF_DATA = True
  MLPERF_REMASK = True
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM

  MESH_SHAPE = [1, 64, 8]

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 1000

  def Task(self) -> InstantiableParams:
    """Returns the task parameters."""
    model_p = super().Task()
    # Enable label smoothing.
    model_p.label_smoothing_prob = 0.1
    return model_p


@model_registry.RegisterModel
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


@model_registry.RegisterModel
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
