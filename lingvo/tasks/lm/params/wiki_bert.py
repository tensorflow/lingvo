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
"""DenseBuilder-based encoder-only BertTransformer."""
from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import gshard_builder
from lingvo.core import optimizer
from lingvo.core import program
from lingvo.core import schedule
from lingvo.tasks.lm import input_generator
import numpy as np

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.tpu import device_assignment as device_assignment_lib
# pylint:enable=g-direct-tensorflow-import


class BertTemplate(base_model_params.SingleTaskModelParams):
  """DenseBuilder-based Bert Transformer Template."""

  # The number of data replicas.
  BATCH_SIZE = 64 * 8
  SEQUENCE_LENGTH = 512

  HIDDEN_DIM = 4096
  ATTENTION_KEY_VALUE_DIM = 128
  MODEL_DIM = 1024
  DROPOUT_RATE = 0.0
  ATTENTION_DROPOUT_RATE = 0.0
  NUM_HEADS = 8
  NUM_TRANSFORMER_LAYERS = 8
  LABEL_SMOOTHING = 0.0
  VOCAB_SIZE = 32000
  # The sharding config of model parallelim
  DEVICE_MESH_SHAPE = [64, 1]  # prod(DEVICE_MESH_SHAPE) = NUM_DEVICES_PER_SPLIT
  DEBUG = False
  DEVICE_MESH = None
  MODEL_DIM_RESHAPE_SEGMENTS = None

  LOSS_DENOMINATOR = 0
  BETA1 = 0
  WARMUP_STEPS = 10000
  TRAIN_STEPS_PER_LOOP = 200
  TRAIN_EXES_PER_EVAL = 5

  POSITIONAL_EMBEDDING = False
  USE_REPEAT_LAYER = False
  GATED_FFN_ACT = 'silu'
  ATTEN_LOGIT_CAP = 0

  def Task(self):
    builder_cls = gshard_builder.DenseBuilder
    p = gshard_builder.BertTransformer.Params().Set(
        use_repeat_layer=self.USE_REPEAT_LAYER,
        gated_ffn_activation=self.GATED_FFN_ACT,
        positional_embedding=self.POSITIONAL_EMBEDDING,
        dtype=tf.float32,
        fprop_dtype=tf.bfloat16,
        name='transformer',
        builder=builder_cls.Params().Set(
            atten_logit_cap=self.ATTEN_LOGIT_CAP,
            attention_num_memory_heads=1,
            device_mesh_shape=self.DEVICE_MESH_SHAPE,
            device_mesh=self.DEVICE_MESH,
            relative_attention_num_buckets=32,
            relative_attention_type='bias',
            relative_attention_max_distance=128,
            dtype=tf.float32,
            fprop_dtype=tf.bfloat16,
            attention_logits_dtype=tf.float32,
            mask_dtype=tf.int32,
            # Using float32 for attention logits with fprop_dtype=bfloat16
            # generally makes training giant models more stable
            dropout_rate=self.DROPOUT_RATE,
            num_devices=1,  # we call .Split num_devices on axis 0 (batch)
            attention_dropout_prob=self.ATTENTION_DROPOUT_RATE,
            attention_key_value_dim=self.ATTENTION_KEY_VALUE_DIM,
            attention_extra_logit=None,
            relative_attention_use_universal_1d_position=True,
            model_dim_reshape_segments=self.MODEL_DIM_RESHAPE_SEGMENTS,
            emb_w_split=[1, 0],
            kv_mhd_w_split=[1, -1, -1],
            emb_out_split=[0, -1, 1],
            blm_split=[0, -1, 1],
            logits_split=[0, -1, 1],
            model_dim=self.MODEL_DIM,
            attention_num_heads=self.NUM_HEADS,
            ff_dim=self.HIDDEN_DIM,
            attention_combine_dims=True),
        batch_size=self.BATCH_SIZE,
        sequence_length=self.SEQUENCE_LENGTH,
        num_transformer_layers=self.NUM_TRANSFORMER_LAYERS,
        aux_loss_coef=0.0,
        loss_denominator=self.LOSS_DENOMINATOR,
        label_smoothing=self.LABEL_SMOOTHING,
        vocab_size=self.VOCAB_SIZE,
        max_length=self.SEQUENCE_LENGTH)

    p.train.tpu_device_order_mode = device_assignment_lib.DeviceOrderMode.MESH

    p.train.optimizer = optimizer.XLAShardingAdafactor.Params().Set(
        beta1=self.BETA1,
        beta2=0.99,
        multiply_by_parameter_scale=True,
        clipping_threshold=1.0,
        factored=True,
        decay_exponent_pow=0.8,
    )

    p.train.learning_rate = 1.0

    p.train.lr_schedule = schedule.SqrtDecay.Params().Set(
        warmup_steps=self.WARMUP_STEPS, multiplier=1.0)

    p.train.max_steps = 10000000
    p.train.save_max_to_keep = 40
    p.train.save_keep_checkpoint_every_n_hours = 12
    p.train.async_checkpointing = True
    return p


class MLPerfTrainTemplate(BertTemplate):
  """Template for MLPerf models."""

  def Task(self):
    p = super().Task()
    p.mask_token_id = 103
    p.masked_lm.mask_token_id = 103
    return p

  def Train(self):
    p = input_generator.TFRecordBertInput.Params()
    p.name = 'train'
    p.resettable = True
    p.batch_size = self.BATCH_SIZE
    p.enable_packing = True
    p.shuffle = True
    p.input_file = 'gs://mlperf_v1_1/bert/train'
    return p

  def Test(self):
    p = input_generator.TFRecordBertInput.Params()
    p.input_file = 'gs://mlperf_v1_1/bert/eval'
    p.name = 'test'
    p.batch_size = 512
    return p

  def ProgramSchedule(self):
    p = program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=self.TRAIN_STEPS_PER_LOOP,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=10,
        decode_steps_per_loop=0,
    )
    p.train_program.spmd = True
    p.train_executions_per_eval = self.TRAIN_EXES_PER_EVAL

    # For compliance logging.
    p.ml_perf.benchmark_name = 'bert'
    p.ml_perf.submission_metadata = {
        'global_batch_size': self.BATCH_SIZE,
        'submission_org': 'Google',
        'submission_platform': 'tpu',
        'submission_division': 'open',
        'submission_status': 'research',
        'submission_benchmark': p.ml_perf.benchmark_name,
        'submission_model': 'lingvo',
        'cache_clear': None,
        'train_samples': 0,
        'eval_samples': 10000
    }

    # For BERT, we log the number of examples as the epoch.
    # epoch_num = global_step / steps_per_epoch
    # epoch_num = num_examples_trained = global_step * examples_per_step
    # steps_per_epoch = global_step / (global_step * examples_per_step)
    # steps_per_epoch = 1 / examples_per_step
    examples_per_step = self.BATCH_SIZE
    p.ml_perf.steps_per_epoch = 1 / examples_per_step

    p.ml_perf.decoder_metric_name = 'acc1'
    p.ml_perf.decoder_metric_success_threshold = 0.6
    p.ml_perf.max_steps_to_train = 31790

    return p


# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.wiki_bert.MLPerfTrainBertDense2B \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=64 \
# --ps_replicas=16 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class MLPerfTrainBertDense2B(MLPerfTrainTemplate):
  """Large Bert model with 2B parameters."""
  VOCAB_SIZE = 30522
  BATCH_SIZE = 1024  # ON 4x4x4

  USE_REPEAT_LAYER = True
  NUM_TRANSFORMER_LAYERS = 8
  MODEL_DIM = 4096
  NUM_HEADS = 16
  HIDDEN_DIM = 16384
  ATTENTION_KEY_VALUE_DIM = 256

  DEVICE_MESH_SHAPE = [16, 4]
  DEVICE_MESH = np.arange(
      0, np.product(DEVICE_MESH_SHAPE)).reshape(DEVICE_MESH_SHAPE)
  MODEL_DIM_RESHAPE_SEGMENTS = [4]


# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.wiki_bert.MLPerfBertDense1T \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=1024 \
# --ps_replicas=256 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class MLPerfBertDense1T(MLPerfTrainTemplate):
  """Large Bert model with 1T parameters on 1024 chips."""
  BATCH_SIZE = 1024

  USE_REPEAT_LAYER = True
  NUM_TRANSFORMER_LAYERS = 128
  HIDDEN_DIM = 131072
  MODEL_DIM = 16384
  NUM_HEADS = 256

  DEVICE_MESH_SHAPE = [64, 16]
  DEVICE_MESH = np.arange(
      0, np.product(DEVICE_MESH_SHAPE)).reshape(DEVICE_MESH_SHAPE)
  HIDDEN_DIM_RESHAPE_SEGMENTS = 16
  MODEL_DIM_RESHAPE_SEGMENTS = [16, 4]


# bazel run -c opt //lingvo:trainer -- --mode=sync \
# --alsologtostderr \
# --model=lm.wiki_bert.MLPerfBertDense1TWider \
# --logdir=${LOGDIR} --tpu=${TPU_NAME} --worker_split_size=1024 \
# --ps_replicas=256 --job=executor_tpu --disable_tf2=true
@model_registry.RegisterSingleTaskModel
class MLPerfBertDense1TWider(MLPerfBertDense1T):
  """Large Bert model with 1T parameters on 1024 chips."""
  BATCH_SIZE = 4096

  NUM_TRANSFORMER_LAYERS = 32
  HIDDEN_DIM = 131072 * 2
  MODEL_DIM = 16384 * 2


@model_registry.RegisterSingleTaskModel
class MLPerfBertDense175B(MLPerfBertDense1T):
  """Large Bert model with 175B parameters on 1024 chips."""
  BATCH_SIZE = 1024
  HIDDEN_DIM = 12288 * 4
  ATTENTION_KEY_VALUE_DIM = 128
  MODEL_DIM = 12288
  NUM_HEADS = 96
  NUM_TRANSFORMER_LAYERS = 96

  POSITIONAL_EMBEDDING = True
  TRAIN_STEPS_PER_LOOP = 20


@model_registry.RegisterSingleTaskModel
class MLPerfBertDense500B(MLPerfBertDense1T):
  """Large Bert model with 481B parameters on 1024 chips."""
  VOCAB_SIZE = 30522
  BATCH_SIZE = 4096

  NUM_TRANSFORMER_LAYERS = 64
  LABEL_SMOOTHING = 0.1

  POSITIONAL_EMBEDDING = True
  REMOVE_MASK = True
  TRAIN_STEPS_PER_LOOP = 100
  TRAIN_EXES_PER_EVAL = 1


@model_registry.RegisterSingleTaskModel
class MLPerfBertDense500B2K(MLPerfBertDense500B):
  """Large Bert model with 481B parameters on 2048 chips."""
  DEVICE_MESH_SHAPE = [256, 8]
  DEVICE_MESH = np.arange(0, np.product(DEVICE_MESH_SHAPE)).reshape(
      [8, 16, 16]).transpose([1, 2, 0]).reshape(DEVICE_MESH_SHAPE)

  HIDDEN_DIM_RESHAPE_SEGMENTS = 8
  MODEL_DIM_RESHAPE_SEGMENTS = [8]
