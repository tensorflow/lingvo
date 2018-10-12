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
"""Encode file using the wpm_encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from lingvo.core import wpm_encoder

tf.flags.DEFINE_string(
    'source_filepaths', '',
    'Comma-separated list of source files. Must align with target')
tf.flags.DEFINE_string('target_filepaths', '',
                       'Comma-separated list of target files.')
tf.flags.DEFINE_string('output_filepath', '', 'File of tfrecords.')
tf.flags.DEFINE_string('wpm_filepath', '', 'The wordpiece vocabulary file.')
tf.flags.DEFINE_integer('num_shards', -1, 'Total number of shards.')
tf.flags.DEFINE_integer('shard_id', -1, 'This shard id (0-based).')
tf.flags.DEFINE_integer(
    'max_len', 0,
    'Drop sentence if src/tgt tokens exceeds max length, including <s> and </s> tags. Only use during training. A value of 0 does not filter'
)

FLAGS = tf.flags.FLAGS

BOW_STR = '\xe2\x96\x81'.decode('utf-8')


def _ReplaceSpaces(text):
  return BOW_STR + text.replace(u' ', BOW_STR)


def _MakeBytesFeature(unicode_array):
  value = [tf.compat.as_bytes(w) for w in unicode_array]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _MakeInt64Feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _MakeFloatFeature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _AssertTextFormat(text):
  assert not text.startswith('<s>')
  assert not text.endswith('</s>')
  assert not text.startswith('<S>')
  assert not text.endswith('</S>')


def _MakeTfExample(enc, source_text, target_text):
  # By convention:
  # * source always ends in </s>, never starts with <s>.
  # * target never ends in </s>, always starts with <s>.
  _AssertTextFormat(source_text)
  _AssertTextFormat(target_text)
  source_text = _ReplaceSpaces(source_text)
  src_s, src_i = zip(*enc.EncodeToStringAndIds(source_text))
  src_s = list(src_s) + [enc.sentence_end_string]
  src_i = list(src_i) + [enc.sentence_end_id]
  if FLAGS.max_len > 0 and len(src_i) > FLAGS.max_len:
    return None
  target_text = _ReplaceSpaces(target_text)
  tgt_s, tgt_i = zip(*enc.EncodeToStringAndIds(target_text))
  tgt_s = [enc.sentence_start_string] + list(tgt_s)
  tgt_l = list(tgt_i) + [enc.sentence_end_id]
  tgt_i = [enc.sentence_start_id] + list(tgt_i)
  if FLAGS.max_len > 0 and len(tgt_i) > FLAGS.max_len:
    return None
  feature = {
      'source_id': _MakeInt64Feature(src_i),
      'source_padding': _MakeFloatFeature(np.zeros_like(src_i)),
      'source_word': _MakeBytesFeature(src_s),
      'target_id': _MakeInt64Feature(tgt_i),
      'target_padding': _MakeFloatFeature(np.zeros_like(tgt_i)),
      'target_word': _MakeBytesFeature(tgt_s),
      'target_label': _MakeInt64Feature(tgt_l),
      'target_weight': _MakeFloatFeature(np.ones_like(tgt_l)),
      'natural_order': _MakeInt64Feature([1]),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _Prepropcess(text):
  if not isinstance(text, unicode):
    text = text.decode('utf-8')
  text = text.strip()
  text = text.replace(' </s>', '')
  return text


def _RunEncoding():
  enc = wpm_encoder.WpmEncoder(FLAGS.wpm_filepath)
  pairs = zip(
      FLAGS.source_filepaths.split(','), FLAGS.target_filepaths.split(','))
  with tf.python_io.TFRecordWriter(FLAGS.output_filepath) as outf:
    n = 0
    for p in pairs:
      with tf.gfile.Open(p[0], 'r') as sourcef:
        with tf.gfile.Open(p[1], 'r') as targetf:
          for textp in zip(sourcef.readlines(), targetf.readlines()):
            n += 1
            if n % 10000 == 0:
              tf.logging.info('Watermark[%d]: %d', FLAGS.shard_id, n)
            if n % FLAGS.num_shards != FLAGS.shard_id:
              continue
            source_text = _Prepropcess(textp[0])
            target_text = _Prepropcess(textp[1])
            # tf.logging.vlog(5, 'Source: %s', source_text)
            # tf.logging.vlog(5, 'Target: %s', target_text)
            ex = _MakeTfExample(enc, _Prepropcess(source_text),
                                _Prepropcess(target_text))
            if not ex:  # Too long.
              continue
            # tf.logging.vlog(5, 'Ex: %s', ex)
            encoded = ex.SerializeToString()
            outf.write(encoded)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  _RunEncoding()


if __name__ == '__main__':
  tf.app.run(main)
