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
from lingvo.core import moe_layers
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

  def Task(self):
    # tokens per batch per replica (~64 cores)
    batch_size_per_tf_replica = int(self.BATCH_DIM_PER_DEVICE *
                                    self.NUM_DEVICES_PER_SPLIT)

    p = gshard_builder.UniTransformer.Params().Set(
        gated_gelu=True,
        debug=self.DEBUG,
        positional_embedding=False,
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
        train_steps_per_loop=100,
        eval_dataset_names=['Train'],
        eval_steps_per_loop=100,
        decode_steps_per_loop=0,
    )
    p.train_program.spmd = True
    # every 5K steps
    p.train_executions_per_eval = 5
    return p


# Total params: 137,702,416,384.
# Expect ~ 3.7k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm128B8x8 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=128 \
# --ps_replicas=8 --cluster_placer_in_executor=true --job=executor_tpu
@model_registry.RegisterSingleTaskModel
class DenseLm128B8x8(DenseLmTemplate):
  """128B params LM model with 2D split."""
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 128
  BATCH_DIM_PER_DEVICE = 0.125
  NUM_TRANSFORMER_LAYERS = 64  # 64 blocks of [DecSelfAttention, DenseReluDense]
  DEVICE_MESH_SHAPE = [8, 16]
  DEVICE_MESH = moe_layers.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [8, 8, 2])

  def Task(self):
    p = super().Task()
    p.train.tpu_device_order_mode = 2  # DeviceOrderMode.MESH
    p.builder.model_dim_reshape_segments = self.DEVICE_MESH_SHAPE[1]
    p.builder.emb_w_split = [-1, 1]
    p.builder.emb_out_split = [0, -1, 1]
    p.builder.blm_split = [0, -1, 1]
    # Partition final logits along B and L so that the argmax will be fully
    # partitioned.
    p.builder.logits_split = [0, 1, -1]
    return p


# Expect ~ 18k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm128B16x16 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=512 \
# --ps_replicas=32 --cluster_placer_in_executor=true --job=executor_tpu
@model_registry.RegisterSingleTaskModel
class DenseLm128B16x16(DenseLm128B8x8):
  """128B params LM model with 2D split on v3-512."""
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 512
  BATCH_DIM_PER_DEVICE = 0.25  # Total batch size 128
  DEVICE_MESH_SHAPE = [16, 32]
  DEVICE_MESH = moe_layers.GetNonPod2dMesh(DEVICE_MESH_SHAPE, [16, 16, 2])


# Total params: 1,100,041,175,040.
# Expect ~ 1.4k tokens/sec
# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr --model=lm.synthetic_packed_input.DenseLm1T16x16 \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=512 \
# --ps_replicas=32 --cluster_placer_in_executor=true --job=executor_tpu
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
# --ps_replicas=128 --cluster_placer_in_executor=true --job=executor_tpu
@model_registry.RegisterSingleTaskModel
class DenseLm128B32x32(DenseLm128B8x8):
  """128B params LM model with 2D split on v3-2048."""
  SEQUENCE_LENGTH = 1024
  NUM_DEVICES_PER_SPLIT = 2048
  BATCH_DIM_PER_DEVICE = 0.25  # Total batch size 512
  DEVICE_MESH_SHAPE = [32, 64]
  DEVICE_MESH = np.reshape(
      np.arange(0, np.product(DEVICE_MESH_SHAPE)), DEVICE_MESH_SHAPE)
