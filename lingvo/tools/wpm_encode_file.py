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
from six import text_type
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
    'Drop sentence if src/tgt tokens exceed max length, counting <s> and </s>. '
    'Only use during training. A value of 0 does not filter.')

FLAGS = tf.flags.FLAGS


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


def _MakeTfExample(enc, src_i, src_s, tgt_i, tgt_s):
  """Creates TfExample from the encoded results."""
  src_i = list(src_i) + [enc.sentence_end_id]
  src_s = list(src_s) + [enc.sentence_end_string]
  if FLAGS.max_len > 0 and len(src_i) > FLAGS.max_len:
    return None
  tgt_l = list(tgt_i) + [enc.sentence_end_id]
  tgt_i = [enc.sentence_start_id] + list(tgt_i)
  tgt_s = [enc.sentence_start_string] + list(tgt_s)
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


def _Preprocess(text):
  if not isinstance(text, text_type):
    text = text.decode('utf-8')
  return text.strip().replace(' </s>', '')


def _RunEncoding():
  sess = tf.Session()
  enc = wpm_encoder.WpmEncoder(FLAGS.wpm_filepath)
  src_txt_placeholder = tf.placeholder(tf.string, [])
  src_encode_op = enc.Encode(src_txt_placeholder)
  tgt_txt_placeholder = tf.placeholder(tf.string, [])
  tgt_encode_op = enc.Encode(tgt_txt_placeholder)
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
            source_text = _Preprocess(textp[0])
            target_text = _Preprocess(textp[1])
            # By convention:
            # * source always ends in </s>, never starts with <s>.
            # * target never ends in </s>, always starts with <s>.
            _AssertTextFormat(source_text)
            _AssertTextFormat(target_text)
            ((src_i, src_s), (tgt_i, tgt_s)) = sess.run(
                [src_encode_op, tgt_encode_op],
                feed_dict={
                    src_txt_placeholder: source_text,
                    tgt_txt_placeholder: target_text
                },
            )
            ex = _MakeTfExample(enc, src_i, src_s, tgt_i, tgt_s)
            if not ex:  # Too long.
              continue
            encoded = ex.SerializeToString()
            outf.write(encoded)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  _RunEncoding()


if __name__ == '__main__':
  tf.app.run(main)
