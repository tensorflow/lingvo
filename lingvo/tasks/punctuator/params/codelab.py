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
from lingvo.tasks.mt.params import base_config
from lingvo.tasks.punctuator import input_generator
from lingvo.tasks.punctuator import model


# This base class defines parameters for the input generator for a specific
# dataset. Specific network architectures will be implemented in subclasses.
class BrownCorpusWPM(base_model_params.SingleTaskModelParams):
  """Brown Corpus data with a Word-Piece Model tokenizer."""

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


# This decorator registers the model in the Lingvo model registry.
# This file is lingvo/tasks/punctuator/params/codelab.py,
# so the model will be registered as punctuator.codelab.RNMTModel.
@model_registry.RegisterSingleTaskModel
class RNMTModel(BrownCorpusWPM):
  """RNMT+ Model."""

  @classmethod
  def Task(cls):
    p = base_config.SetupRNMTParams(
        model.RNMTModel.Params(),
        name='punctuator_rnmt',
        vocab_size=cls._VOCAB_SIZE,
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
    p.eval.samples_per_summary = 2466
    return p
