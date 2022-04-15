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
# ==============================================================================
"""Downloads, builds, and preprocesses the COCO captions dataset.

The tool generates an augmented version of COCO captions in which each example
stores precomputed BERT embeddings of the caption text. This augmented dataset
can be used with Milan "BERT adapter" models like the Crisscrossed Captions dual
encoder (see cxc.py).

NOTE: This tool downloads data and models from the web (via
`tensorflow_datasets` and `tensorflow_hub`) and altogether may consume 50+ GB of
storage. Users can set the `TFDS_DATA_DIR` and `TFHUB_CACHE_DIR` environment
variables to control where the intermediate data is stored and/or to reuse
cached copies.

Example usage:
$ prepare_coco --splits=train,dev,test --output_dir=/my/output/directory
"""

import os
from typing import Dict, Iterator

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from lingvo import compat as tf
from lingvo.core import base_layer
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_text  # pylint: disable=unused-import

flags.DEFINE_list('splits', 'train,dev,test', 'Data splits to prepare.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer(
    'num_output_shards', 50,
    'Number of ways to shard the output tfrecords (per split).')
flags.DEFINE_integer('num_workers', 4,
                     'Number of parallel worker threads to use.')

flags.DEFINE_string(
    'bert_tfhub_model',
    'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'URL/path of the tf-hub BERT model.')
flags.DEFINE_string(
    'bert_tfhub_preprocessor',
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'URL/path of the tf-hub BERT preprocessor.')
flags.DEFINE_integer('bert_max_length', 48,
                     'Max token length of extracted BERT embeddings.')

flags.mark_flag_as_required('output_dir')

FLAGS = flags.FLAGS


def log(message):
  logging.info(message)
  print(message)


class TfHubBertEncoder(base_layer.BaseLayer):
  """Encodes text inputs using a pretrained BERT model from TF-hub."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('preprocessor', '', 'URL/path of the tf-hub BERT preprocessor.')
    p.Define('model', '', 'URL/path of the tf-hub BERT model.')
    p.Define('max_seq_len', 48,
             'Maximum encoded sequence length (including CLS and SEP).')
    p.name = 'bert_encoder'
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    log(f'Initializing BERT preprocessor {p.preprocessor}.')
    preprocessor = hub.load(p.preprocessor)
    self._tokenizer = hub.KerasLayer(preprocessor.tokenize)
    self._input_packer = hub.KerasLayer(
        preprocessor.bert_pack_inputs, arguments=dict(seq_length=p.max_seq_len))
    log(f'Initializing BERT model {p.model}.')
    self._model = hub.KerasLayer(p.model)

  def FProp(self, _, input_strings):
    """Tokenizes and encodes the given strings.

    Args:
      input_strings: Text strings to encode.

    Returns:
      A tuple (padded_features, num_tokens) where
        * padded_features is a float Tensor of token-level embeddings, shape
          [batch_size, max_seq_len, bert_feature_dim]. (The features are
          zero-padded out to max_seq_len tokens.)
        * lengths is an int Tensor holding the token length of each input.
          Slice `padded_features[i, :lengths[i], :]` holds the actual token
          embeddings for the i'th input.
    """
    bert_inputs = self._input_packer([self._tokenizer(input_strings)])
    sequence_features = self._model(bert_inputs)['sequence_output']
    sequence_lengths = tf.reduce_sum(bert_inputs['input_mask'], axis=1)
    return sequence_features, sequence_lengths


ExampleDict = Dict[str, np.ndarray]


def dict_to_example_proto(example_dict: ExampleDict) -> tf.train.Example:
  """Converts a `dict`-format example to a `tf.train.Example` proto."""
  example = tf.train.Example()
  for name, value in example_dict.items():
    value = np.reshape(np.asarray(value), [-1])

    # Find the type-specific value list where this feature should be stored.
    dest_feature = example.features.feature[name]
    dtype = tf.dtypes.as_dtype(value.dtype)
    if dtype.is_floating:
      dest_feature.float_list.value[:] = value
    elif dtype.is_integer:
      dest_feature.int64_list.value[:] = value
    elif dtype == tf.string:
      # In Python3 strings must be explicitly encoded into bytes.
      if isinstance(value[0], str):
        value = [each.encode() for each in value]
      dest_feature.bytes_list.value[:] = value
    else:
      raise ValueError('Unsupported feature dtype {}'.format(dtype))

  return example


class ReadCocoFromTfdsFn(beam.DoFn):
  """Reads splits of the 'coco_captions' dataset via `tensorflow_datasets`."""

  def process(self, split: str) -> Iterator[ExampleDict]:
    dataset = tfds.load(
        'coco_captions',
        split=split,
        decoders={'image': tfds.decode.SkipDecoding()})
    for example in dataset.as_numpy_iterator():
      yield {
          'image/encoded': example['image'],
          'image/id': example['image/id'],
          'text/captions': example['captions']['text'],
          'text/id': example['captions']['id'],
      }


class EncodeCaptionsFn(beam.DoFn):
  """Generates encodings of text captions."""

  def __init__(self, encoder_params):
    self._encoder_params = encoder_params

  def setup(self):
    super().setup()
    self._encoder = self._encoder_params.Instantiate()

  def process(self, example: ExampleDict) -> Iterator[ExampleDict]:
    captions = example['text/captions']
    assert captions.ndim == 1
    features, lengths = self._encoder(captions)

    # Flatten each N-caption example to N single-caption examples.
    for i in range(features.shape[0]):
      output_example = example.copy()
      output_example.update({
          'text/bert/token_features': features[i],
          'text/bert/lengths': lengths[i]
      })
      yield output_example


def main(_):
  log('Loading "coco_captions" from tfds. This will build a copy of '
      'the dataset under TFDS_DATA_DIR if one doesn\'t already exist.')
  tfds.load('coco_captions')

  # Milan split name => constituent tfds subsplits. (Subsplitting enables the
  # tfds data to be processed in parallel.)
  split_to_coco_subsplits = dict(
      train=tfds.even_splits('train', 100) + tfds.even_splits('restval', 100),
      dev=tfds.even_splits('val', 50),
      test=tfds.even_splits('test', 50))

  encoder_params = TfHubBertEncoder.Params().Set(
      preprocessor=FLAGS.bert_tfhub_preprocessor,
      model=FLAGS.bert_tfhub_model,
      max_seq_len=FLAGS.bert_max_length)

  def pipeline(root):
    for split in FLAGS.splits:
      subsplits = split_to_coco_subsplits[split]
      _ = root | split.title() >> (
          'Create' >> beam.Create(subsplits)
          | 'Read' >> beam.ParDo(ReadCocoFromTfdsFn())
          | 'Encode' >> beam.ParDo(EncodeCaptionsFn(encoder_params))
          | 'ToProto' >> beam.Map(dict_to_example_proto)
          | 'Shuffle' >> beam.Reshuffle()
          | 'Write' >> beam.io.tfrecordio.WriteToTFRecord(
              file_path_prefix=os.path.join(FLAGS.output_dir, split),
              coder=beam.coders.ProtoCoder(tf.train.Example),
              num_shards=FLAGS.num_output_shards))

  pipeline_options = beam.options.pipeline_options.PipelineOptions()
  pipeline_options.view_as(beam.options.pipeline_options.DirectOptions
                          ).direct_num_workers = FLAGS.num_workers
  with beam.Pipeline(options=pipeline_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
