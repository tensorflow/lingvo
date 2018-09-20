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
"""Several functions to initialize typical values of dataset parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from lingvo.core import layers
from lingvo.core import lr_schedule
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.tasks.mt import decoder
from lingvo.tasks.mt import encoder
from lingvo.tasks.mt import input_generator
from lingvo.tasks.mt import model


def InitTrainDatasetParams(vocab_size=None, params=None):
  """Initializes typical values for train datasets.

  Args:
    vocab_size: the number of tokens in your vocabulary. The default is None
      because this parameter is often not used.
    params: initial Params value, e.g. `NmtInput.Params()`.

  Returns:
    a `Params` object.
  """
  if params is None:
    params = input_generator.NmtInput.Params()
  params.is_nmt_example = True

  params.file_random_seed = 0

  # How many threads to run in parallel.
  params.file_parallelism = 16

  # Note, for training, we prefer to use big file_buffer_size (as long as all
  # fits in RAM), to more thoroughly randomize the training examples. when the
  # file_buffer_size too small, we run the risk of sequentially going over the
  # example as they are stored in the sstable which may not be random (e.g.
  # maybe alphabetically ordered).
  params.file_buffer_size = 10000000

  if vocab_size is not None:
    params.tokenizer.vocab_size = vocab_size

  # The bucket upper bound is determined based on an exponentially growing
  # scheme, with _GenerateBuckets(10, 100) resulting buckets starting from
  # minimum bucket size of 10 to maximum bucket size of 137.
  # For word and sub-word level NMT, we train on sequences up to maximum
  # bucket size and discard the examples that are longer than 137.
  params.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137]

  # The bucket batch limit determines how many examples are there in each
  # batch during training. We reduce the batch size for the buckets that
  # have higher upper bound (batches that consist of longer sequences eg.
  # 98, 137) in order to prevent out of memory issues.
  # Note that this hyperparameter varies widely based on the model and language.
  # larger models may warrant smaller batches in order to fit in memory, for
  # example; and ideographical languages like Chinese may benefit from more
  # buckets.
  params.bucket_batch_limit = [128] * 8 + [64]
  return params


def InitTestDatasetParams(vocab_size=None, params=None):
  """Initializes typical values for test and dev datasets.

  Args:
    vocab_size: the number of tokens in your vocabulary.
    params: initial Params value, e.g. `NmtInput.Params()`.

  Returns:
    a `Params` object.
  """

  if params is None:
    params = input_generator.NmtInput.Params()

  params.file_random_seed = 27182818

  # How many threads to run in parallel.
  params.file_parallelism = 1

  # In order to make exactly one pass over the dev/test sets, we set buffer
  # size to 1. Greater numbers may cause inaccurate dev/test scores.
  params.file_buffer_size = 1

  if vocab_size is not None:
    params.tokenizer.vocab_size = vocab_size

  # The largest bucket upper bound must be larger than the longest sequence
  # length in dev/test set. Since we discard sequences longer than the
  # max(bucket_upper_bound) we may end up having scores based on only shorter
  # sequences only if we mistakenly set this to be too small.
  params.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
  params.bucket_batch_limit = [128] * 8 + [64] + [32]
  return params


def InitTransformerTestBuckets(params):
  params.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 137, 200]
  params.bucket_batch_limit = [16] * 10
  return params


def InitTransformerTrainBuckets(params):
  params.bucket_upper_bound = [8, 12, 16, 24, 32, 48, 64, 96]
  params.bucket_batch_limit = [512, 341, 256, 170, 128, 85, 64, 42]
  return params


def SetupTransformerParams(name,
                           vocab_size,
                           model_dim,
                           hidden_dim,
                           num_heads,
                           num_layers,
                           learning_rate,
                           warmup_steps,
                           residual_dropout_prob=0.1,
                           input_dropout_prob=0.0,
                           atten_dropout_prob=0.0,
                           relu_dropout_prob=0.0,
                           label_smoothing_uncertainty=0.1,
                           is_transparent=False):
  """Common model setup for different transformer models.

  Args:
    name: An identifier for an instance of a transformer model.
    vocab_size: an integer representing the size of the vocabulary, probably
         16000 or 32000.
    model_dim: dimension of the transformer block (column)
    hidden_dim: dimension of Feed-Forward neural network in each layer
    num_heads: number of attention heads to use for the transformer
    num_layers: number of layers in the transformer
    learning_rate: learning rate for Adam. For the base model, we use 1.0; for
         the big model, 3.0
    warmup_steps: warmup steps for `TransformerLearningRateSchedule`. For the
         base model, we use 4000; for the big model, 40000
    residual_dropout_prob: dropout prob to the output of each sub-layer before
         it is added to the sub-layer input
    input_dropout_prob: dropout prob to the sums of the token embeddings and the
         position embeddings
    atten_dropout_prob: dropout prob to the attention weights in each
         Transformer attention sub-layer
    relu_dropout_prob: dropout prob to the inner layer output (ReLU activation)
         in each Transformer feed-forward sub-layer
    label_smoothing_uncertainty: if this value is 0, no label smoothing will be
         applied
    is_transparent: If set, decoder layers attend to weighted combinations of
        encoder layers.

  Returns:
    A Params object containing the parameters that specify a transformer model
    (Vaswani 2017)

  """
  p = model.TransformerModel.Params()
  p.name = name

  # Transformer encoder and decoder setup
  p.encoder = SetupTransformerEncoder(
      model_dim, vocab_size, num_layers, num_heads, hidden_dim,
      residual_dropout_prob, input_dropout_prob, atten_dropout_prob,
      relu_dropout_prob, is_transparent)
  p.decoder = SetupTransformerDecoder(
      model_dim, vocab_size, num_layers, num_heads, hidden_dim,
      residual_dropout_prob, input_dropout_prob, atten_dropout_prob,
      relu_dropout_prob, label_smoothing_uncertainty, is_transparent)

  p.train.Set(
      learning_rate=learning_rate,
      optimizer=optimizer.Adam.ParamsB(),
      clip_gradient_norm_to_value=0.0,
      grad_norm_to_clip_to_zero=0.0,
      lr_schedule=lr_schedule.TransformerLearningRateSchedule.Params().Set(
          warmup_steps=warmup_steps, worker_replicas=1, model_dim=model_dim))

  p.eval.samples_per_summary = 12000
  return p


def SetupTransformerDecoder(model_dim,
                            vocab_size,
                            num_layers,
                            num_heads,
                            hidden_dim,
                            residual_dropout_prob=0.1,
                            input_dropout_prob=0.0,
                            atten_dropout_prob=0.0,
                            relu_dropout_prob=0.0,
                            label_smoothing_uncertainty=0.1,
                            is_transparent=False):
  """Common setup for transformer model decoder."""
  disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
  default_params_init = py_utils.WeightInit.Xavier(1.0)
  emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

  # Decoder
  decoder_params = decoder.TransformerDecoder.Params()
  decoder_params.source_dim = model_dim
  decoder_params.model_dim = model_dim
  decoder_params.num_trans_layers = num_layers
  decoder_params.input_dropout_prob = input_dropout_prob

  decoder_params.token_emb.Set(
      vocab_size=vocab_size,
      embedding_dim=model_dim,
      max_num_shards=16,
      params_init=emb_params_init,
      vn=disable_vn,
      scale_sqrt_depth=True)

  decoder_params.position_emb.Set(
      embedding_dim=model_dim, trainable_scaling=False, vn=disable_vn)

  decoder_params.trans_tpl.source_dim = model_dim
  decoder_params.trans_tpl.tr_atten_tpl.Set(
      source_dim=model_dim,
      num_attention_heads=num_heads,
      residual_dropout_prob=residual_dropout_prob,
      atten_dropout_prob=atten_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn)

  decoder_params.trans_tpl.tr_atten_tpl.atten_tpl.Set(
      enable_ctx_pre_proj=True,
      enable_ctx_post_proj=True,
      context_dim=model_dim,
      vn=disable_vn)

  decoder_params.trans_tpl.tr_fflayer_tpl.Set(
      input_dim=model_dim,
      hidden_dim=hidden_dim,
      residual_dropout_prob=residual_dropout_prob,
      relu_dropout_prob=relu_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn)

  decoder_params.softmax.Set(
      num_classes=vocab_size,
      vn=disable_vn,
      params_init=emb_params_init,
      num_shards=16)

  decoder_params.per_word_avg_loss = True
  decoder_params.label_smoothing = layers.UniformLabelSmoother.Params()
  decoder_params.label_smoothing.num_classes = vocab_size
  decoder_params.label_smoothing.uncertainty = label_smoothing_uncertainty

  if is_transparent:
    decoder_params.is_transparent = True

  return decoder_params


def SetupTransformerEncoder(model_dim,
                            vocab_size,
                            num_layers,
                            num_heads,
                            hidden_dim,
                            residual_dropout_prob=0.1,
                            input_dropout_prob=0.0,
                            atten_dropout_prob=0.0,
                            relu_dropout_prob=0.0,
                            is_transparent=False):
  """Common setup for transformer model encoder.

  Args:
   model_dim: specifies dimension of transformer layers, token embeddings,
    and positional embeddings as well context vectors (attention values).
   vocab_size: for token embeddings.
   num_layers: number of transformer layers.
   num_heads: number of attention heads.
   hidden_dim: in transformer feedforward layer.
   residual_dropout_prob: used in transformer feedforward and attention layer.
   input_dropout_prob: input dropout.
   atten_dropout_prob: used in attention layer.
   relu_dropout_prob: used in transformer feedforward layer.
   is_transparent: if set, outputs a merger of embeddings and layer outputs.

  Returns:
   Encoder params.
  """
  disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
  default_params_init = py_utils.WeightInit.Xavier(1.0)
  emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

  # Encoder
  encoder_params = encoder.TransformerEncoder.Params()

  encoder_params.token_emb.Set(
      embedding_dim=model_dim,
      max_num_shards=16,
      params_init=emb_params_init,
      vocab_size=vocab_size,
      vn=disable_vn,
      scale_sqrt_depth=True)

  encoder_params.position_emb.Set(
      embedding_dim=model_dim, trainable_scaling=False, vn=disable_vn)

  # Encoder TransformerStack params
  encoder_params.model_dim = model_dim
  encoder_params.transformer_stack.model_dim = model_dim
  encoder_params.transformer_stack.num_transformer_layers = num_layers
  encoder_params.input_dropout_prob = input_dropout_prob

  encoder_params.transformer_stack.transformer_tpl.tr_atten_tpl.Set(
      num_attention_heads=num_heads,
      residual_dropout_prob=residual_dropout_prob,
      atten_dropout_prob=atten_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn)

  encoder_params.transformer_stack.transformer_tpl.tr_atten_tpl.atten_tpl.Set(
      num_attention_heads=num_heads,
      enable_ctx_pre_proj=True,
      enable_ctx_post_proj=True,
      context_dim=model_dim,
      vn=disable_vn)

  encoder_params.transformer_stack.transformer_tpl.tr_fflayer_tpl.Set(
      hidden_dim=hidden_dim,
      residual_dropout_prob=residual_dropout_prob,
      relu_dropout_prob=relu_dropout_prob,
      params_init=default_params_init,
      vn=disable_vn)

  if is_transparent:
    encoder_params.transformer_stack.is_transparent = True

  return encoder_params
