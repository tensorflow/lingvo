# Lint as: python2, python3
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
"""Models for Librispeech dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import datasource
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import tokenizers
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import model


@model_registry.RegisterSingleTaskModel
class Librispeech960Base(base_model_params.SingleTaskModelParams):
  """Base parameters for Librispeech 960 hour task."""

  def _CommonInputParams(self, is_eval):
    """Input generator params for Librispeech."""
    p = input_generator.AsrInput.Params()

    # Insert path to the base directory where the data are stored here.
    # Generated using scripts in lingvo/tasks/asr/tools.
    p.file_datasource = datasource.PrefixedDataSourceWrapper.Params()
    p.file_datasource.base_datasource = datasource.SimpleDataSource.Params()
    p.file_datasource.base_datasource.file_type = 'tfrecord'
    p.file_datasource.file_pattern_prefix = '/tmp/librispeech'

    p.frame_size = 80
    p.append_eos_frame = True

    p.pad_to_max_seq_length = False
    p.file_random_seed = 0
    p.file_buffer_size = 10000
    p.file_parallelism = 16

    p.is_eval = is_eval

    if is_eval:
      p.source_max_length = 3600
      p.bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 3600]
    else:
      p.source_max_length = 3000
      p.bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 1710]

    p.bucket_batch_limit = [96, 48, 48, 48, 48, 48, 48, 48]

    return p

  def SetBucketSizes(self, params, bucket_upper_bound, bucket_batch_limit):
    """Sets bucket sizes for batches in params."""
    params.bucket_upper_bound = bucket_upper_bound
    params.bucket_batch_limit = bucket_batch_limit
    return params

  def Train(self):
    p = self._CommonInputParams(is_eval=False)
    p.file_datasource.base_datasource.file_pattern = 'train/train.tfrecords-*'
    p.num_samples = 281241
    return p

  def Dev(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.base_datasource.file_pattern = (
        'devtest/dev-clean.tfrecords-00000-of-00001')
    p.num_samples = 2703
    return p

  def Devother(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.base_datasource.file_pattern = (
        'devtest/dev-other.tfrecords-00000-of-00001')
    p.num_samples = 2864
    return p

  def Test(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.base_datasource.file_pattern = (
        'devtest/test-clean.tfrecords-00000-of-00001')
    p.num_samples = 2620
    return p

  def Testother(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.base_datasource.file_pattern = (
        'devtest/test-other.tfrecords-00000-of-00001')
    p.num_samples = 2939
    return p

  def Task(self):
    p = model.AsrModel.Params()
    p.name = 'librispeech'

    # Initialize encoder params.
    ep = p.encoder
    # Data consists 240 dimensional frames (80 x 3 frames), which we
    # re-interpret as individual 80 dimensional frames. See also,
    # LibrispeechCommonAsrInputParams.
    ep.input_shape = [None, None, 80, 1]
    ep.lstm_cell_size = 1024
    ep.num_lstm_layers = 4
    ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)]
    ep.conv_filter_strides = [(2, 2), (2, 2)]
    ep.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.001)
    # Disable conv LSTM layers.
    ep.num_conv_lstm_layers = 0

    # Initialize decoder params.
    dp = p.decoder
    dp.rnn_cell_dim = 1024
    dp.rnn_layers = 2
    dp.source_dim = 2048
    # Use functional while based unrolling.
    dp.use_while_loop_based_unrolling = False

    tp = p.train
    tp.learning_rate = 2.5e-4
    tp.lr_schedule = schedule.ContinuousLearningRateSchedule.Params().Set(
        start_step=50000, half_life_steps=100000, min=0.01)

    # Setting p.eval.samples_per_summary to a large value ensures that dev,
    # devother, test, testother are evaluated completely (since num_samples for
    # each of these sets is less than 5000), while train summaries will be
    # computed on 5000 examples.
    p.eval.samples_per_summary = 5000
    p.eval.decoder_samples_per_summary = 0

    # Use variational weight noise to prevent overfitting.
    p.vn.global_vn = True
    p.train.vn_std = 0.075
    p.train.vn_start_step = 20000

    return p

  def ProgramSchedule(self):
    return program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=50,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=5,
        decode_steps_per_loop=0)


@model_registry.RegisterSingleTaskModel
class Librispeech960Grapheme(Librispeech960Base):
  """Base params for Librispeech 960 hour experiments using grapheme models.

  With 8 workers using asynchronous gradient descent on 16 (8x2) GPUs, the model
  achieves the following error rates after ~853.2K steps:

  ========= =====
  Dev       5.2%
  DevOther  15.2%
  Test      5.4%
  TestOther 15.5%
  ========= =====
  """

  GRAPHEME_TARGET_SEQUENCE_LENGTH = 620
  GRAPHEME_VOCAB_SIZE = 76

  def InitializeTokenizer(self, params):
    """Initializes a grapheme tokenizer."""
    params.tokenizer = tokenizers.AsciiTokenizer.Params()
    tokp = params.tokenizer
    tokp.vocab_size = self.GRAPHEME_VOCAB_SIZE
    tokp.append_eos = True
    tokp.target_unk_id = 0
    tokp.target_sos_id = 1
    tokp.target_eos_id = 2

    params.target_max_length = self.GRAPHEME_TARGET_SEQUENCE_LENGTH
    return params

  def Train(self):
    p = super(Librispeech960Grapheme, self).Train()
    return self.InitializeTokenizer(params=p)

  def Dev(self):
    p = super(Librispeech960Grapheme, self).Dev()
    return self.InitializeTokenizer(params=p)

  def Devother(self):
    p = super(Librispeech960Grapheme, self).Devother()
    return self.InitializeTokenizer(params=p)

  def Test(self):
    p = super(Librispeech960Grapheme, self).Test()
    return self.InitializeTokenizer(params=p)

  def Testother(self):
    p = super(Librispeech960Grapheme, self).Testother()
    return self.InitializeTokenizer(params=p)

  def Task(self):
    p = super(Librispeech960Grapheme, self).Task()
    dp = p.decoder
    dp.target_seq_len = self.GRAPHEME_TARGET_SEQUENCE_LENGTH
    dp.emb_dim = self.GRAPHEME_VOCAB_SIZE
    dp.emb.vocab_size = self.GRAPHEME_VOCAB_SIZE
    dp.softmax.num_classes = self.GRAPHEME_VOCAB_SIZE
    return p


@model_registry.RegisterSingleTaskModel
class Librispeech960GraphemeTpuV2(Librispeech960Grapheme):
  """Librispeech 960 grapheme model for training on TPU V2."""

  def _CommonInputParams(self, is_eval):
    p = super(Librispeech960GraphemeTpuV2, self)._CommonInputParams(is_eval)

    p.pad_to_max_seq_length = True
    p.bucket_batch_limit = [48] * len(p.bucket_upper_bound)

    p.source_max_length = p.bucket_upper_bound[-1]\

    return p

  def Task(self):
    p = super(Librispeech960GraphemeTpuV2, self).Task()

    p.encoder.pad_steps = 0

    return p


@model_registry.RegisterSingleTaskModel
class Librispeech960Wpm(Librispeech960Base):
  """Base params for Librispeech 960 hour experiments using Word Piece Models.

  With 8 workers using asynchronous gradient descent on 16 (8x2) GPUs, the model
  achieves the following error rates after ~632.6K steps:

  ========= =====
  Dev       4.3%
  DevOther  13.0%
  Test      4.5%
  TestOther 13.2%
  ========= =====
  """

  # Set this to a WPM vocabulary file before training. By default, we use the
  # pre-generated 16K word piece vocabulary checked in under 'tasks/asr/'.
  WPM_SYMBOL_TABLE_FILEPATH = (
      'lingvo/tasks/asr/wpm_16k_librispeech.vocab')
  WPM_TARGET_SEQUENCE_LENGTH = 140
  WPM_VOCAB_SIZE = 16328

  EMBEDDING_DIMENSION = 96
  NUM_TRAINING_WORKERS = 8

  def InitializeTokenizer(self, params):
    """Initializes a Word Piece Tokenizer."""
    params.tokenizer = tokenizers.WpmTokenizer.Params()
    tokp = params.tokenizer
    tokp.vocab_filepath = self.WPM_SYMBOL_TABLE_FILEPATH
    tokp.vocab_size = self.WPM_VOCAB_SIZE
    tokp.append_eos = True
    tokp.target_unk_id = 0
    tokp.target_sos_id = 1
    tokp.target_eos_id = 2

    params.target_max_length = self.WPM_TARGET_SEQUENCE_LENGTH
    return params

  def Train(self):
    p = super(Librispeech960Wpm, self).Train()
    return self.InitializeTokenizer(params=p)

  def Dev(self):
    p = super(Librispeech960Wpm, self).Dev()
    return self.InitializeTokenizer(params=p)

  def Devother(self):
    p = super(Librispeech960Wpm, self).Devother()
    return self.InitializeTokenizer(params=p)

  def Test(self):
    p = super(Librispeech960Wpm, self).Test()
    return self.InitializeTokenizer(params=p)

  def Testother(self):
    p = super(Librispeech960Wpm, self).Testother()
    return self.InitializeTokenizer(params=p)

  def Task(self):
    p = super(Librispeech960Wpm, self).Task()
    dp = p.decoder
    dp.target_seq_len = self.WPM_TARGET_SEQUENCE_LENGTH
    dp.emb_dim = self.EMBEDDING_DIMENSION
    dp.emb.vocab_size = self.WPM_VOCAB_SIZE
    dp.emb.max_num_shards = self.NUM_TRAINING_WORKERS  # One shard per worker.
    dp.softmax.num_classes = self.WPM_VOCAB_SIZE

    return p


@model_registry.RegisterSingleTaskModel
class Librispeech960WpmTpuV2(Librispeech960Wpm):
  """Librispeech 960 WPM model for training on TPU V2."""

  def _CommonInputParams(self, is_eval):
    p = super(Librispeech960WpmTpuV2, self)._CommonInputParams(is_eval)

    p.pad_to_max_seq_length = True
    p.bucket_batch_limit = [48] * len(p.bucket_upper_bound)
    p.source_max_length = p.bucket_upper_bound[-1]

    return p

  def Task(self):
    p = super(Librispeech960WpmTpuV2, self).Task()

    p.encoder.pad_steps = 0
    p.decoder.emb.max_num_shards = 1

    return p
