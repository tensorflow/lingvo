# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""DenseBuilder-based LM with synthetic inputs."""

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import base_input_generator
from lingvo.core import base_model_params
from lingvo.core import gshard_builder
from lingvo.core import gshard_utils
from lingvo.core import optimizer
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import schedule
import numpy as np


class SyntheticTrain(base_input_generator.BaseInputGenerator):
  """Generated synthetic data with packed_input lm formats."""

  @classmethod
  def Params(cls):
    """Defaults params for input generators."""
    p = super().Params()
    p.Define('seq_len', 0, 'Number of tokens in one example')
    return p

  def _InputBatch(self):
    seq_len = self.params.seq_len
    targets = tf.ones([self.params.batch_size, seq_len], dtype=tf.int32)
    input_batch = py_utils.NestedMap()
    input_batch.tgt = py_utils.NestedMap()
    input_batch.tgt.ids = tf.roll(targets, 1, axis=1)
    input_batch.tgt.labels = targets
    input_batch.tgt.segment_ids = tf.minimum(targets, 1)
    input_batch.tgt.segment_pos = targets
    input_batch = input_batch.Transform(
        lambda t: tf.ensure_shape(t, (self.params.batch_size, seq_len)))
    return input_batch


class DenseLmTemplate(base_model_params.SingleTaskModelParams):
  """DenseBuilder-based LM Template."""
  # Batch size per replica = BATCH_DIM_PER_DEVICE * NUM_DEVICES_PER_SPLIT
  BATCH_DIM_PER_DEVICE = 0.0625
  NUM_DEVICES_PER_SPLIT = 64  # number of devices per data replica.
  SEQUENCE_LENGTH = 1024

  HIDDEN_DIM = 65536
  ATTENTION_KEY_VALUE_DIM = 128
  MODEL_DIM = 8192
  NUM_HEADS = 128
  NUM_TRANSFORMER_LAYERS = 32
  LABEL_SMOOTHING = 0.0
  VOCAB_SIZE = 32000
  # The sharding config of model parallelim
  DEVICE_MESH_SHAPE = [64, 1]  # prod(DEVICE_MESH_SHAPE) = NUM_DEVICES_PER_SPLIT
  DEVICE_MESH = None
  DEBUG = False

  GATED_GELU = True
  POSITIONAL_EMBEDDING = False
  USE_REPEAT_LAYER = False
  TRAIN_STEPS_PER_LOOP = 100

  def Task(self):
    # tokens per batch per replica (~64 cores)
    batch_size_per_tf_replica = int(self.BATCH_DIM_PER_DEVICE *
                                    self.NUM_DEVICES_PER_SPLIT)

    p = gshard_builder.UniTransformer.Params().Set(
        gated_gelu=self.GATED_GELU,
        debug=self.DEBUG,
        positional_embedding=self.POSITIONAL_EMBEDDING,
        use_repeat_layer=self.USE_REPEAT_LAYER,
        dtype=tf.float32,
        fprop_dtype=tf.bfloat16,
        name='transformer',
        builder=gshard_builder.DenseBuilder.Params().Set(
            device_mesh_shape=self.DEVICE_MESH_SHAPE,
            device_mesh=self.DEVICE_MESH,
            relative_attention_num_buckets=32,
            relative_attention_type='bias',
            relative_attention_max_distance=128,
            dtype=tf.float32,
            fprop_dtype=tf.bfloat16,
            dropout_rate=0.0,
            num_devices=1,  # Obsolete params
            attention_dropout_prob=0.0,
            attention_key_value_dim=self.ATTENTION_KEY_VALUE_DIM,
            attention_extra_logit=0.0,
            relative_attention_use_universal_1d_position=True,
            model_dim=self.MODEL_DIM,
            attention_num_heads=self.NUM_HEADS,
            ff_dim=self.HIDDEN_DIM,
            attention_combine_dims=True),
        batch_size=batch_size_per_tf_replica,
        sequence_length=self.SEQUENCE_LENGTH,
        num_transformer_layers=self.NUM_TRANSFORMER_LAYERS,
        aux_loss_coef=0.0,
        label_smoothing=self.LABEL_SMOOTHING,
        vocab_size=self.VOCAB_SIZE,
        max_length=self.SEQUENCE_LENGTH)

    p.train.optimizer = optimizer.XLAShardingAdafactor.Params().Set(
        beta1=0.0,
        beta2=0.99,
        multiply_by_parameter_scale=True,
        clipping_threshold=1.0,
        factored=True,
        decay_exponent_pow=0.8,
    )

    p.train.learning_rate = 1.0

    p.train.lr_schedule = schedule.SqrtDecay.Params().Set(
        warmup_steps=10000, multiplier=1.0)

    p.train.max_steps = 2000000
    p.train.save_max_to_keep = 100

    return p

  def Train(self):
    p = SyntheticTrain.Params()
    p.batch_size = int(self.BATCH_DIM_PER_DEVICE * self.NUM_DEVICES_PER_SPLIT)
    p.seq_len = self.SEQUENCE_LENGTH
    return p

  def ProgramSchedule(self):
    p = program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=self.TRAIN_STEPS_PER_LOOP,
        eval_dataset_names=[],
        eval_steps_per_loop=0,
        decode_steps_per_loop=0,
    )
    p.train_program.spmd = True
    # every 5K steps
    p.train_executions_per_eval = 5
    return p


@model_registry.RegisterSingleTaskModel
class DenseLm8B2x2(DenseLmTemplate):
  """8B params LM model with 1D split."""
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 128
  BATCH_DIM_PER_DEVICE = 0.125
  NUM_TRANSFORMER_LAYERS = 4  # 4 blocks of [DecSelfAttention, DenseReluDense]
  DEVICE_MESH_SHAPE = [1, 8]
  DEVICE_MESH = np.arange(8).reshape(DEVICE_MESH_SHAPE)

  def Task(self):
    p = super().Task()
    p.train.tpu_device_order_mode = 2  # DeviceOrderMode.MESH
    p.builder.model_dim_reshape_segments = self.DEVICE_MESH_SHAPE[1]
    p.builder.emb_w_split = [-1, 1]
    p.builder.emb_out_split = [0, -1, 1]
    p.builder.blm_split = [0, -1, 1]
    p.builder.logits_split = [0, -1, 1]
    return p


@model_registry.RegisterSingleTaskModel
class DenseLm8B2x2Decode(DenseLm8B2x2):
  """8B params LM decoding config."""

  def Task(self):
    p = super().Task()
    # relative_attention_use_universal_1d_position should be set to False in
    # decoding.
    p.builder.relative_attention_use_universal_1d_position = False
    return p


# Total params: 137,702,416,384.
# Expect ~ 3.7k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm128B8x8 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=128 \
# --ps_replicas=16 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm128B8x8(DenseLmTemplate):
  """128B params LM model with 2D split."""
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 128
  BATCH_DIM_PER_DEVICE = 0.125
  NUM_TRANSFORMER_LAYERS = 64  # 64 blocks of [DecSelfAttention, DenseReluDense]
  DEVICE_MESH_SHAPE = [8, 16]
  DEVICE_MESH = gshard_utils.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [8, 8, 2])

  def Task(self):
    p = super().Task()
    p.train.tpu_device_order_mode = 2  # DeviceOrderMode.MESH
    p.builder.model_dim_reshape_segments = self.DEVICE_MESH_SHAPE[1]
    p.builder.emb_w_split = [-1, 1]
    p.builder.emb_out_split = [0, -1, 1]
    p.builder.blm_split = [0, -1, 1]
    p.builder.logits_split = [0, -1, 1]
    return p


# Expect ~ 18k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm128B16x16 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=512 \
# --ps_replicas=64 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm128B16x16(DenseLm128B8x8):
  """128B params LM model with 2D split on v3-512."""
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 512
  BATCH_DIM_PER_DEVICE = 0.25  # Total batch size 128
  DEVICE_MESH_SHAPE = [16, 32]
  DEVICE_MESH = gshard_utils.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [16, 16, 2])


# Total params: 174366928896
# Expect ~51.53k tokens / sec
@model_registry.RegisterSingleTaskModel
class DenseLm175B32x32(DenseLm128B16x16):
  """175B params LM model with 2D split on v3-2048."""
  HIDDEN_DIM = 12288 * 4
  ATTENTION_KEY_VALUE_DIM = 128
  MODEL_DIM = 12288
  NUM_HEADS = 96
  NUM_TRANSFORMER_LAYERS = 96
  GATED_GELU = False
  POSITIONAL_EMBEDDING = True
  USE_REPEAT_LAYER = True

  SEQUENCE_LENGTH = 2048
  NUM_DEVICES_PER_SPLIT = 2048
  BATCH_DIM_PER_DEVICE = 0.5  # Total batch size 2M tokens
  DEVICE_MESH_SHAPE = [64, 32]
  DEVICE_MESH = np.reshape(
      np.arange(0, np.product(DEVICE_MESH_SHAPE)), [32, 64]).transpose()


@model_registry.RegisterSingleTaskModel
class DenseLm175B8x8Decode2D(DenseLm175B32x32):
  """175B params LM model decoding on v3-128.

  2D logical mesh. It can load a checkpoint from DenseLm175B32x32.
  """
  BATCH_DIM_PER_DEVICE = 0.125
  NUM_DEVICES_PER_SPLIT = 128
  # NUM_HEADS is not a multiple of 128 so we use 2D sharding on M and H.
  DEVICE_MESH_SHAPE = [8, 16]
  DEVICE_MESH = gshard_utils.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [8, 8, 2])

  def Task(self):
    p = super().Task()
    # relative_attention_use_universal_1d_position should be set to False in
    # decoding.
    p.builder.relative_attention_use_universal_1d_position = False
    p.builder.model_dim_reshape_segments = self.DEVICE_MESH_SHAPE[0]
    p.builder.emb_w_split = [1, 0]
    p.builder.emb_out_split = [-1, -1, 0]
    p.builder.blm_split = [-1, -1, 0]
    p.builder.blh_split = [-1, -1, 1]
    p.builder.qkv_split = [0, -1, 1, -1]  # [-1, -1, 1, -1] for global batch 1.
    p.builder.logits_split = [-1, -1, 1]
    return p


# Total params: 1,100,041,175,040.
# Expect ~ 1.4k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm1T16x16 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=512 \
# --ps_replicas=64 --job=executor_tpu
@model_registry.RegisterSingleTaskModel
class DenseLm1T16x16(DenseLm128B16x16):
  """1T params LM model with 2D split on v3-512."""
  SEQUENCE_LENGTH = 512
  BATCH_DIM_PER_DEVICE = 0.03125  # Total batch size 16
  NUM_TRANSFORMER_LAYERS = 128
  HIDDEN_DIM = 131072
  MODEL_DIM = 16384
  NUM_HEADS = 256


# Expect ~ 62k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm128B32x32 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=2048 \
# --ps_replicas=256 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm128B32x32(DenseLm128B8x8):
  """128B params LM model with 2D split on v3-2048."""
  TRAIN_STEPS_PER_LOOP = 20
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 2048
  BATCH_DIM_PER_DEVICE = 0.25  # Total batch size 512
  DEVICE_MESH_SHAPE = [32, 64]
  DEVICE_MESH = np.reshape(
      np.arange(0, np.product(DEVICE_MESH_SHAPE)), DEVICE_MESH_SHAPE)


class ShardedAdamOptimizer(tf.train.AdamOptimizer):
  """Adam optimizer that shards the slot variables."""

  def _create_slots(self, var_list):
    super()._create_slots(var_list)

    for var in var_list:
      try:
        sharding = gshard_utils.GetVarSharding(var)
      except ValueError:
        continue
      if sharding.is_replicated:
        continue
      m = self.get_slot(var, 'm')
      v = self.get_slot(var, 'v')
      sharding.ApplyToVariable(m)
      sharding.ApplyToVariable(v)


class ShardedAdam(optimizer.Adam):
  """Adam optimizer wrapper that shards the slot variables."""

  @classmethod
  def Params(cls):
    params = super().Params()
    params.Define('num_micro_batches', 1, 'Number of accumulated batches.')
    return params

  def GetOptimizer(self, lr):
    p = self.params
    opt = ShardedAdamOptimizer(
        learning_rate=lr,
        beta1=p.beta1,
        beta2=p.beta2,
        epsilon=p.epsilon,
        name=p.name)
    if p.num_micro_batches > 1:
      tf.logging.info('Applying gradient aggregation.')
      opt = optimizer.GradientAggregationOptimizer(
          opt, p.num_micro_batches, apply_crs_to_grad=True)
    return opt


# Expect ~ 53.8k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.synthetic_packed_input.DenseLm12kWide41BAdam16x16 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=512 \
# --ps_replicas=64 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm12kWide41BAdam16x16(DenseLm128B16x16):
  """41B params LM model with 2D split and ADAM optimizer on v3-512."""

  # Each layer has 1.6875B parameters.
  SEQUENCE_LENGTH = 2048
  NUM_DEVICES_PER_SPLIT = 512
  BATCH_DIM_PER_DEVICE = 0.5  # Total batch size 256
  DEVICE_MESH_SHAPE = [16, 32]
  DEVICE_MESH = gshard_utils.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [16, 16, 2])
  NUM_TRANSFORMER_LAYERS = 24
  HIDDEN_DIM = 48 * 1024
  MODEL_DIM = 12 * 1024
  NUM_HEADS = 96
  ATTENTION_KEY_VALUE_DIM = 128
  GATED_GELU = False
  POSITIONAL_EMBEDDING = True
  NUM_MICRO_BATCHES = 1

  def Task(self):
    p = super().Task()
    p.train.optimizer = ShardedAdam.Params().Set(
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        num_micro_batches=self.NUM_MICRO_BATCHES)
    return p


# Expect ~ 17.4k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.synthetic_packed_input.DenseLm12kWide10BAdam8x8 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=128 \
# --ps_replicas=16 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm12kWide41BAdam8x8(DenseLm12kWide41BAdam16x16):
  # IF OOM, try 0.25 BATCH_DIM_PER_DEVICE and 8 NUM_MICRO_BATCHES
  BATCH_DIM_PER_DEVICE = 0.5  # Total micro-batch size 64
  NUM_MICRO_BATCHES = 4  # Total batch size 256
  NUM_DEVICES_PER_SPLIT = 128
  DEVICE_MESH_SHAPE = [8, 16]
  DEVICE_MESH = gshard_utils.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [8, 8, 2])


# Expect ~ 12.5k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.synthetic_packed_input.DenseLm12kWide162BAdam16x16 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=512 \
# --ps_replicas=64 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm12kWide162BAdam16x16(DenseLm12kWide41BAdam16x16):
  """162B params LM model with 2D split and ADAM optimizer on v3-512."""

  BATCH_DIM_PER_DEVICE = 0.125  # Total batch size 64
  NUM_TRANSFORMER_LAYERS = 96
  DEVICE_MESH_SHAPE = [16, 32]
  DEVICE_MESH = gshard_utils.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [16, 16, 2])


@model_registry.RegisterSingleTaskModel
class DenseLm12kWide162BAdamBS25616x16(DenseLm12kWide162BAdam16x16):
  BATCH_DIM_PER_DEVICE = 0.125  # Total micro batch size 64
  NUM_MICRO_BATCHES = 4  # Total batch size 256


# Expect ~ XXX tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.synthetic_packed_input.DenseLm12kWide162BAdam32x32 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=2048 \
# --ps_replicas=256 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class DenseLm12kWide162BAdam32x32(DenseLm12kWide162BAdam16x16):
  """162B params LM model with 2D split and ADAM optimizer on v3-2048."""
  TRAIN_STEPS_PER_LOOP = 20
  NUM_DEVICES_PER_SPLIT = 2048
  BATCH_DIM_PER_DEVICE = 0.125  # Total batch size 256
  # NUM_HEADS is 96, so we shard it 32 ways.
  DEVICE_MESH_SHAPE = [64, 32]
  DEVICE_MESH = np.reshape(
      np.arange(0, np.product(DEVICE_MESH_SHAPE)), [32, 64]).transpose()
