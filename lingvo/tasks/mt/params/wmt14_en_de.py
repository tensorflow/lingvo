# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Train NMT Models on WMT'14 English-German machine translation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.tasks.mt import input_generator
from lingvo.tasks.mt import model
from lingvo.tasks.mt.params import base_config


@model_registry.RegisterSingleTaskModel
class WmtEnDeTransformerBase(base_model_params.SingleTaskModelParams):
  """Params for WMT'14 En->De."""

  DATADIR = '/tmp/wmt14/wpm/'
  VOCAB_SIZE = 32000

  @classmethod
  def Train(cls):
    p = input_generator.NmtInput.Params()

    p.file_random_seed = 0
    p.file_parallelism = 16
    p.file_buffer_size = 10000000

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR,
                                                'train.tfrecords-*')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, 'wpm-ende.voc')

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 4492447
    p.bucket_upper_bound = (
        [8, 10, 12, 14, 16, 20, 24, 28] + [32, 40, 48, 56, 64, 80, 96])
    p.bucket_batch_limit = ([512, 409, 341, 292, 256, 204, 170, 146] +
                            [128, 102, 85, 73, 64, 51, 42])
    return p

  @classmethod
  def Dev(cls):
    p = input_generator.NmtInput.Params()
    p.file_random_seed = 27182818
    p.file_parallelism = 1
    p.file_buffer_size = 1

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR, 'dev.tfrecords')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, 'wpm-ende.voc')

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 3000
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
    p.bucket_batch_limit = [16] * 8 + [4] * 2
    return p

  @classmethod
  def Test(cls):
    p = input_generator.NmtInput.Params()
    p.file_random_seed = 27182818
    p.file_parallelism = 1
    p.file_buffer_size = 1

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR, 'test.tfrecords')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, 'wpm-ende.voc')

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 2737
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
    p.bucket_batch_limit = [16] * 8 + [4] * 2
    return p

  @classmethod
  def Task(cls):
    p = base_config.SetupTransformerParams(
        model.TransformerModel.Params(),
        name='wmt14_en_de_transformer_base',
        vocab_size=cls.VOCAB_SIZE,
        model_dim=512,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6,
        residual_dropout_prob=0.1,
        input_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=40000)
    p.eval.samples_per_summary = 7500
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeTransformerSmall(WmtEnDeTransformerBase):
  """Small Transformer Params for WMT'14 En->De."""

  @classmethod
  def Task(cls):
    p = base_config.SetupTransformerParams(
        model.TransformerModel.Params(),
        name='wmt14_en_de_transformer_small',
        vocab_size=cls.VOCAB_SIZE,
        model_dim=64,
        hidden_dim=128,
        num_heads=2,
        num_layers=2,
        residual_dropout_prob=0.1,
        input_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=40000)
    p.eval.samples_per_summary = 7500
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeRNMT(base_model_params.SingleTaskModelParams):
  """Params for WMT'14 En->De in sync training."""

  # Generated using scripts in lingvo/mt/tools.
  DATADIR = '/tmp/wmt14/wpm/'
  VOCAB_SIZE = 32000

  @classmethod
  def Train(cls):
    p = input_generator.NmtInput.Params()

    p.file_random_seed = 0
    p.file_parallelism = 16
    p.file_buffer_size = 10000000

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR,
                                                'train.tfrecords-*')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, 'wpm-ende.voc')

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 4492447
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98]
    p.bucket_batch_limit = [128] * 7 + [64]
    return p

  @classmethod
  def Dev(cls):
    p = input_generator.NmtInput.Params()
    p.file_random_seed = 27182818
    p.file_parallelism = 1
    p.file_buffer_size = 1

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR, 'dev.tfrecords')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, 'wpm-ende.voc')

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 3000
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  @classmethod
  def Test(cls):
    p = input_generator.NmtInput.Params()
    p.file_random_seed = 27182818
    p.file_parallelism = 1
    p.file_buffer_size = 1

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR, 'test.tfrecords')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, 'wpm-ende.voc')

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 2737
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  @classmethod
  def Task(cls):
    p = base_config.SetupRNMTParams(
        model.RNMTModel.Params(),
        name='wmt14_en_de_rnmtplus_base',
        vocab_size=cls.VOCAB_SIZE,
        embedding_dim=1024,
        hidden_dim=1024,
        num_heads=4,
        num_encoder_layers=6,
        num_decoder_layers=8,
        learning_rate=1e-4,
        l2_regularizer_weight=1e-5,
        lr_warmup_steps=500,
        lr_decay_start=400000,
        lr_decay_end=1200000,
        lr_min=0.5,
        ls_uncertainty=0.1,
        atten_dropout_prob=0.3,
        residual_dropout_prob=0.3,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
    )
    p.eval.samples_per_summary = 7500
    return p
