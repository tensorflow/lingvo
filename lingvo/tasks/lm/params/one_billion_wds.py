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
"""Train word-level LMs on 1 Billion Words benchmark data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import layers
from lingvo.core import lr_schedule
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import tokenizers
from lingvo.tasks.lm import input_generator as lm_inp
from lingvo.tasks.lm import layers as lm_layers
from lingvo.tasks.lm import model


class WordLevelOneBwdsBase(base_model_params.SingleTaskModelParams):
  """Params for training a word-level LM on One Billion Wds text corpus.

  Tries to match https://github.com/rafaljozefowicz/lm.
  """

  # Generated using lingvo/tasks/lm/tools:download_lm1b.
  CORPUS_DIR = '/tmp/lm1b/1-billion-word-language-modeling-benchmark-r13output/'

  # BIG-LSTM model size: embedding/projection dim = 1024; LSTM state dim = 8192
  EMBEDDING_DIM = 1024
  MAX_TOKENS = 1024
  NUM_EMBEDDING_SHARDS = 8
  NUM_SAMPLED = 8192
  NUM_SOFTMAX_SHARDS = 8
  RNN_STATE_DIM = 8192
  VOCAB_SIZE = 793472  # includes <epsilon>
  WORD_VOCAB = os.path.join(CORPUS_DIR, 'vocab.txt')

  @classmethod
  def Train(cls):
    p = lm_inp.LmInput.Params()
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [1024, 512, 256, 256, 128, 128, 64, 32, 16]
    p.file_buffer_size = 10000000
    p.file_parallelism = 10
    p.file_pattern = 'text:' + os.path.join(
        cls.CORPUS_DIR, 'training-monolingual.tokenized.shuffled', 'news.en*')
    p.name = '1bwds_train_set'
    p.tokenizer = tokenizers.VocabFileTokenizer.Params()
    p.num_batcher_threads = 16
    p.target_max_length = cls.MAX_TOKENS
    p.tokenizer.target_sos_id = 1
    p.tokenizer.target_eos_id = 2
    p.tokenizer.target_unk_id = 3
    p.tokenizer.token_vocab_filepath = cls.WORD_VOCAB
    p.tokenizer.vocab_size = cls.VOCAB_SIZE
    return p

  @classmethod
  def Dev(cls):
    p = cls.Train()
    # Use small batches for eval.
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 64, 32, 32, 16, 16, 4, 2, 1]
    p.file_buffer_size = 1
    p.file_parallelism = 1
    p.file_pattern = 'text:' + os.path.join(
        cls.CORPUS_DIR, 'heldout-monolingual.tokenized.shuffled',
        'news.en.heldout-00001*')
    p.name = '1bwds_dev_set'
    p.num_batcher_threads = 1
    p.num_samples = 6206  # Number of sentences to evaluate on.
    return p

  @classmethod
  def Test(cls):
    p = cls.Dev()
    p.file_pattern = 'text:' + os.path.join(
        cls.CORPUS_DIR, 'heldout-monolingual.tokenized.shuffled',
        'news.en.heldout-00000*')
    p.name = '1bwds_test_set'
    p.num_samples = 6075  # Number of sentences to evaluate on.
    return p

  @classmethod
  def Task(cls):
    p = model.LanguageModel.Params()
    p.name = '1bwds_word_level_lm'
    p.eval.samples_per_summary = 10000

    p.lm = lm_layers.RnnLm.CommonParams(
        vocab_size=cls.VOCAB_SIZE,
        emb_dim=cls.EMBEDDING_DIM,
        num_layers=2,
        residual_start=3,  # disable residuals
        rnn_dims=cls.EMBEDDING_DIM,
        rnn_hidden_dims=cls.RNN_STATE_DIM)

    # Input embedding needs to be sharded.
    p.lm.emb.max_num_shards = cls.NUM_EMBEDDING_SHARDS
    p.lm.embedding_dropout_keep_prob = 0.75
    # Match the initialization in github code.
    p.lm.emb.params_init = py_utils.WeightInit.UniformUnitScaling(
        1.0 * cls.NUM_EMBEDDING_SHARDS)

    # We also want dropout after each of the RNN layers.
    p.lm.rnns.dropout.keep_prob = 0.75

    # Adjusts training params.
    tp = p.train
    tp.sum_loss_across_tokens_in_batch = True
    # Disable any so called "clipping" (gradient scaling really).
    tp.clip_gradient_norm_to_value = 0.0
    tp.grad_norm_to_clip_to_zero = 0.0
    # Do clip the LSTM gradients.
    tp.max_lstm_gradient_norm = 16
    # Straight Adagrad; very sensitive to initial accumulator value, the default
    # 0.1 value is far from adequate.
    # TODO(ciprianchelba): tune accumulator value, learning rate, clipping
    # threshold.
    tp.learning_rate = 0.2
    tp.lr_schedule = (
        lr_schedule.PiecewiseConstantLearningRateSchedule.Params().Set(
            boundaries=[], values=[1.0]))
    tp.l2_regularizer_weight = None  # No regularization.
    tp.optimizer = optimizer.Adagrad.Params()
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelOneBwdsSimpleSampledSoftmax(WordLevelOneBwdsBase):
  """Use sampled soft-max in training."""

  @classmethod
  def Task(cls):
    p = super(WordLevelOneBwdsSimpleSampledSoftmax, cls).Task()
    num_input_dim = p.lm.softmax.input_dim
    p.lm.softmax = layers.SimpleFullSoftmax.Params()
    p.lm.softmax.input_dim = num_input_dim
    p.lm.softmax.num_classes = cls.VOCAB_SIZE
    p.lm.softmax.num_sampled = cls.NUM_SAMPLED
    p.lm.softmax.num_shards = cls.NUM_SOFTMAX_SHARDS
    # Match the initialization in github code.
    p.lm.softmax.params_init = py_utils.WeightInit.UniformUnitScaling(
        1.0 * cls.NUM_SOFTMAX_SHARDS)
    assert p.lm.softmax.num_classes % p.lm.softmax.num_shards == 0
    return p


@model_registry.RegisterSingleTaskModel
class WordLevelOneBwdsSimpleSampledSoftmaxTiny(
    WordLevelOneBwdsSimpleSampledSoftmax):
  """Tiny model size for local, debugging runs of the above."""

  EMBEDDING_DIM = 7
  MAX_TOKENS = 1024
  NUM_EMBEDDING_SHARDS = 1
  NUM_SAMPLED = 8
  NUM_SOFTMAX_SHARDS = 8
  RNN_STATE_DIM = 32


# Word-level log-pplx on eval_test: 0.02@13.4k
# Try the following params and deploy 8 accelerators to train a bigger model
# LAYERS = 94  #4.9B params
# MAX_SEQLEN = 1024
# SPLITS = [10, 22, 34, 46, 58, 70, 82, 94]  # On 8 accelerator, 16G mem each.
# BATCH_SIZE = 32
# NUM_MICRO_BATCHES = 32
@model_registry.RegisterSingleTaskModel
class OneBWdsGPipeTransformer(WordLevelOneBwdsBase):
  """LM using gpipe transformer."""
  VOCAB_SIZE = 32000
  EMBEDDING_DIM = 2048
  BATCH_SIZE = 8
  MAX_SEQLEN = 100
  LAYERS = 6
  # GPIPE related params.
  SPLITS = 1
  NUM_MICRO_BATCHES = 1

  @classmethod
  def Train(cls):
    p = super(OneBWdsGPipeTransformer, cls).Train()
    p.bucket_upper_bound = [cls.MAX_SEQLEN]
    p.bucket_batch_limit = [cls.BATCH_SIZE]
    p.fixed_input_shape = True
    return p

  @classmethod
  def Task(cls):
    """Language model on 1bw dataset using gpipe transformer."""
    p = model.FixedShapeInputLanguageModel.Params()
    p.eval.samples_per_summary = 0
    p.name = '1bwds_wpm_level_lm'
    p.lm = lm_layers.GPipeTransformerLm.CommonParams(
        model_dim=cls.EMBEDDING_DIM,
        vocab_size=cls.VOCAB_SIZE,
        hidden_dim=cls.EMBEDDING_DIM * 4,
        num_layers=cls.LAYERS,
        splits=cls.SPLITS,
        num_micro_batches=cls.NUM_MICRO_BATCHES,
        num_heads=16,
        softmax_max_alloc=128 * (2**20),
        atten_dropout_prob=0.1,
        residual_dropout_prob=0.1)

    p.train.Set(
        learning_rate=0.5,
        optimizer=optimizer.Adam.ParamsA(),
        clip_gradient_norm_to_value=0.0,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=lr_schedule.TransformerLearningRateSchedule.Params().Set(
            warmup_steps=40000, worker_replicas=1, model_dim=cls.EMBEDDING_DIM))
    return p
