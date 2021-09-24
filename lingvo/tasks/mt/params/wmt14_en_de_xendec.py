# Lint as: python3
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

import os

from lingvo import compat as tf
from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import py_utils
from lingvo.tasks.mt import base_config
from lingvo.tasks.mt import input_generator
from lingvo.tasks.mt import model


@model_registry.RegisterSingleTaskModel
class WmtEnDeXEnDec(base_model_params.SingleTaskModelParams):
  """Params for WMT'14 En->De experiments in https://arxiv.org/abs/2106.04060."""

  DATADIR = '/tmp/wmt14ende/'
  DATATRAIN = 'tmp-*'
  DATADEV = 'tmp-000-010'
  DATATEST = 'tmp-000-010'
  VOCAB = 'wordpiece-mixed.vocab'
  PACKED_INPUT = True
  vocab_size = 32000

  add_unnormalized_residuals = True
  pre_layer_norm = True
  residual_dropout_prob = 0.1
  input_dropout_prob = 0.1
  atten_dropout_prob = 0.
  relu_dropout_prob = 0.
  num_heads = 8
  model_dim = 512
  hidden_dim = 2048

  source_mask_ratio = 0.
  source_mask_ratio_beta = '2,6'
  mask_word_id = 3
  pad_id = 4
  mask_words_ratio = 0.25
  permutation_distance = 3
  loss_mix_weight = 1.0
  loss_clean_weight = 1.0
  loss_mono_weight = 1.0

  batch_size_ratio = 1
  learning_rate = 1.0
  num_samples = 4506303
  use_prob_cl = True
  use_atten_drop = True
  use_atten_cl = True
  use_prob_drop = False

  def Train(self):
    p = input_generator.NmtDoubleInput.Params()
    p.file_random_seed = 0
    p.file_parallelism = 64
    p.file_buffer_size = 10000000
    p.bucket_upper_bound = [
        8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
        160, 192, 224, 256
    ]
    p.bucket_batch_limit = [
        512, 409, 341, 292, 256, 204, 170, 146, 128, 102, 85, 73, 64, 51, 42,
        36, 32, 25, 21, 18, 16
    ]
    p.bucket_batch_limit = [
        max(int(a * self.batch_size_ratio), 1) for a in p.bucket_batch_limit
    ]

    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16

    p.file_pattern = 'tfrecord:' + os.path.join(self.DATADIR, self.DATATRAIN)
    p.tokenizer.token_vocab_filepath = os.path.join(self.DATADIR, self.VOCAB)
    p.natural_order_model = True
    p.num_samples = self.num_samples

    p.source_mask_ratio = self.source_mask_ratio
    p.source_mask_ratio_beta = self.source_mask_ratio_beta
    p.mask_word_id = self.mask_word_id
    p.pad_id = self.pad_id
    p.mask_words_ratio = self.mask_words_ratio
    p.permutation_distance = self.permutation_distance
    p.packed_input = self.PACKED_INPUT
    p.vocab_file = os.path.join(self.DATADIR, self.VOCAB)
    return p

  def _EvalParams(self):
    """Input generator params for WMT'14 En->De."""
    p = input_generator.NmtInput.Params()
    p.tokenizer.vocab_size = self.vocab_size
    p.file_random_seed = 27182818
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
    p.bucket_batch_limit = [16] * 8 + [4] * 2
    return p

  def Dev(self):
    """newstest2013 is used for development."""

    p = self._EvalParams()
    p.file_pattern = 'tfrecord:' + os.path.join(self.DATADIR, self.DATADEV)
    p.tokenizer.token_vocab_filepath = os.path.join(self.DATADIR, self.VOCAB)
    p.num_samples = 3000
    return p

  def Test(self):
    """newstest2014 is used for test."""

    p = self._EvalParams()
    p.file_pattern = 'tfrecord:' + os.path.join(self.DATADIR, self.DATATEST)
    p.tokenizer.token_vocab_filepath = os.path.join(self.DATADIR, self.VOCAB)
    p.num_samples = 3003
    return p

  def Task(self):
    p = model.TransformerXEnDecModel.Params()
    p = base_config.SetupXEnDecTransformerParams(
        p,
        name='transformer',
        vocab_size=self.vocab_size,
        model_dim=self.model_dim,
        hidden_dim=self.hidden_dim,
        num_heads=self.num_heads,
        num_layers=6,
        residual_dropout_prob=self.residual_dropout_prob,
        input_dropout_prob=self.input_dropout_prob,
        atten_dropout_prob=self.atten_dropout_prob,
        relu_dropout_prob=self.relu_dropout_prob,
        learning_rate=self.learning_rate,
        warmup_steps=4000)

    p.loss_mix_weight = self.loss_mix_weight
    p.loss_clean_weight = self.loss_clean_weight
    p.loss_mono_weight = self.loss_mono_weight
    p.use_prob_cl = self.use_prob_cl
    p.use_atten_drop = self.use_atten_drop
    p.decoder.use_atten_cl = self.use_atten_cl
    p.use_prob_drop = self.use_prob_drop

    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
      p.train.tpu_steps_per_loop = 1000
      for pp in [p.encoder, p.decoder]:
        pp.packed_input = self.PACKED_INPUT

    p.train.save_keep_checkpoint_every_n_hours = 1. / 6
    p.decoder.beam_search.length_normalization = 0.6
    p.decoder.beam_search.beam_size = 4
    return p
