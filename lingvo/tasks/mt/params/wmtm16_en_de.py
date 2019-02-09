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
"""Train NMT Models on WMT'16 MMT English-German machine translation task."""

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
class WmtCaptionEnDeTransformer(base_model_params.SingleTaskModelParams):
  """Params for WMT'16 En->De Captions (ignoring the images)."""

  # Generated using scripts in lingvo/mt/tools.
  DATADIR = '/tmp/wmtm16/wpm/'
  VOCAB_SIZE = 2000
  VOCAB_FILE = 'wpm-ende-2k.voc'

  @classmethod
  def Train(cls):
    p = input_generator.NmtInput.Params()

    p.file_random_seed = 0
    p.file_parallelism = 1
    p.file_buffer_size = 29000

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR, 'train.tfrecords')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, cls.VOCAB_FILE)

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 29000
    p.bucket_upper_bound = [14, 17, 20, 24, 29, 35, 45, 75]
    p.bucket_batch_limit = [292, 240, 204, 170, 141, 117, 91, 54]
    return p

  @classmethod
  def Dev(cls):
    p = input_generator.NmtInput.Params()
    p.file_random_seed = 27182818
    p.file_parallelism = 1
    p.file_buffer_size = 1

    p.file_pattern = 'tfrecord:' + os.path.join(cls.DATADIR, 'val.tfrecords')
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, cls.VOCAB_FILE)

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 1014
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
    p.tokenizer.token_vocab_filepath = os.path.join(cls.DATADIR, cls.VOCAB_FILE)

    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    p.num_samples = 1000
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
    p.bucket_batch_limit = [16] * 8 + [4] * 2
    return p

  @classmethod
  def Task(cls):
    p = base_config.SetupTransformerParams(
        model.TransformerModel.Params(),
        name='wmt14_en_de_transformer_base',
        vocab_size=cls.VOCAB_SIZE,
        model_dim=256,
        hidden_dim=512,
        num_heads=2,
        num_layers=2,
        residual_dropout_prob=0.2,
        input_dropout_prob=0.2,
        learning_rate=1.0,
        warmup_steps=1000)
    p.eval.samples_per_summary = 7500
    p.train.save_interval_seconds = 60
    p.train.max_steps = 12000
    return p
