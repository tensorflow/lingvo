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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tensorflow as tf

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import lr_schedule
from lingvo.core import optimizer
from lingvo.tasks.mt.params import base_config
from lingvo.tasks.punctuator import input_generator
from lingvo.tasks.punctuator import model


# This decorator registers the model in the Lingvo model registry.
# This file is lingvo/tasks/punctuator/params/codelab.py,
# so the model will be registered as punctuator.codelab.TransformerModel.
@model_registry.RegisterSingleTaskModel
class TransformerModel(base_model_params.SingleTaskModelParams):
  """A transformer model for the punctuator task."""

  # Generated using
  # lingvo/tasks/punctuator/tools:download_brown_corpus.
  _DATADIR = '/tmp/punctuator_data'
  _VOCAB_FILE = tf.resource_loader.get_path_to_datafile(
      'brown_corpus_wpm.16000.vocab')
  # _VOCAB_SIZE needs to be a multiple of 16 because we use a sharded softmax
  # with 16 shards.
  _VOCAB_SIZE = 16000

  @classmethod
  def Train(cls):
    p = input_generator.PunctuatorInput.Params()
    p.file_pattern = 'text:' + os.path.join(cls._DATADIR, 'train.txt')
    p.file_random_seed = 0  # Do not use a fixed seed.
    p.file_parallelism = 1  # We only have a single input file.

    # The bucket upper bound specifies how to split the input into buckets. We
    # train on sequences up to maximum bucket size and discard longer examples.
    p.bucket_upper_bound = [10, 20, 30, 60, 120]

    # The bucket batch limit determines how many examples are there in each
    # batch during training. We reduce the batch size for the buckets that
    # have higher upper bound (batches that consist of longer sequences)
    # in order to prevent out of memory issues.
    # Note that this hyperparameter varies widely based on the model and
    # language. Larger models may warrant smaller batches in order to fit in
    # memory, for example; and ideographical languages like Chinese may benefit
    # from more buckets.
    p.bucket_batch_limit = [512, 256, 160, 80, 40]

    p.tokenizer.vocab_filepath = cls._VOCAB_FILE
    p.tokenizer.vocab_size = cls._VOCAB_SIZE
    p.tokenizer.pad_to_max_length = False

    # Set the tokenizer max length slightly longer than the largest bucket to
    # discard examples that are longer than we allow.
    p.source_max_length = p.bucket_upper_bound[-1] + 2
    p.target_max_length = p.bucket_upper_bound[-1] + 2
    return p

  # There is also a Dev method for dev set params, but we don't have a dev set.
  @classmethod
  def Test(cls):
    p = input_generator.PunctuatorInput.Params()
    p.file_pattern = 'text:' + os.path.join(cls._DATADIR, 'test.txt')
    p.file_random_seed = 27182818  # Fix random seed for testing.
    # The following two parameters are important if there's more than one input
    # file. For this codelab it doesn't actually matter.
    p.file_parallelism = 1  # Avoid randomness in testing.
    # In order to make exactly one pass over the dev/test sets, we set buffer
    # size to 1. Greater numbers may cause inaccurate dev/test scores.
    p.file_buffer_size = 1

    p.bucket_upper_bound = [10, 20, 30, 60, 120, 200]
    p.bucket_batch_limit = [16] * 4 + [4] * 2

    p.tokenizer.vocab_filepath = cls._VOCAB_FILE
    p.tokenizer.vocab_size = cls._VOCAB_SIZE
    p.tokenizer.pad_to_max_length = False

    p.source_max_length = p.bucket_upper_bound[-1] + 2
    p.target_max_length = p.bucket_upper_bound[-1] + 2
    return p

  @classmethod
  def Task(cls):
    p = model.TransformerModel.Params()
    p.name = 'punctuator_transformer'

    model_dim = 512
    vocab_size = cls._VOCAB_SIZE
    num_layers = 6
    num_heads = 8
    hidden_dim = 2048
    residual_dropout_prob = 0.1
    input_dropout_prob = 0.1

    # Transformer encoder and decoder setup, delegated to
    # lingvo/tasks/mt/params/base_config.py.
    p.encoder = base_config.SetupTransformerEncoder(
        model_dim, vocab_size, num_layers, num_heads, hidden_dim,
        residual_dropout_prob, input_dropout_prob)
    p.decoder = base_config.SetupTransformerDecoder(
        model_dim, vocab_size, num_layers, num_heads, hidden_dim,
        residual_dropout_prob, input_dropout_prob)

    tp = p.train
    tp.learning_rate = 3.0
    tp.optimizer = optimizer.Adam.ParamsB()
    tp.clip_gradient_norm_to_value = 0.0
    tp.grad_norm_to_clip_to_zero = 0.0
    tp.lr_schedule = lr_schedule.TransformerLearningRateSchedule.Params().Set(
        warmup_steps=40000, worker_replicas=1, model_dim=model_dim)
    return p
