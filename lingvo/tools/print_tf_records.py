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
"""Debug print tf records in text format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf

tf.flags.DEFINE_string('input_filepattern', '',
                       'File pattern of binary tfrecord files.')
tf.flags.DEFINE_string('input_format', 'tf.Example',
                       'Input format: only "tf.Example" supported for now.')
tf.flags.DEFINE_integer('skip_first_n', 0, 'Skip first records.')
tf.flags.DEFINE_integer('print_only_n', -1,
                        'Only print a certain number of records.')
tf.flags.DEFINE_bool('abbreviated', True, 'Print in abbreviated format.')
tf.flags.DEFINE_bool('bytes_as_utf8', True,
                     'Print byte strings as UTF-8 strings')
tf.flags.DEFINE_bool('count_only', False,
                     'Don\'t print, just count number of entries')

FLAGS = tf.flags.FLAGS


def _ListDebugString(values, to_string=str):
  if len(values) <= 8:
    return repr(values)
  first_values = [to_string(v) for v in values[0:6]]
  last_values = [to_string(v) for v in values[-2:]]
  return '[' + ' '.join(first_values + ['...'] + last_values) + ']'


def _CustomShortDebugString(tf_example):
  text = []
  for name, value in sorted(six.iteritems(tf_example.features.feature)):
    if value.HasField('bytes_list'):
      if FLAGS.bytes_as_utf8:
        utf8_values = [v.decode('utf-8') for v in value.bytes_list.value]
        value_string = _ListDebugString(utf8_values)
      else:
        value_string = _ListDebugString(value.bytes_list.value)
    elif value.HasField('float_list'):
      value_string = _ListDebugString(value.float_list.value)
    elif value.HasField('int64_list'):
      value_string = _ListDebugString(value.int64_list.value, to_string=repr)
    text += ['%s: %s' % (name, value_string)]
  return '\n'.join(text)


def _PrintHeader(tf_example):
  """Prints table of contents."""
  # Typically, tf.Examples have the same features.
  tf.logging.info('==== FEATURES ====')
  for name, value in sorted(six.iteritems(tf_example.features.feature)):
    type_string = '<empty>'
    if value.HasField('bytes_list'):
      type_string = 'bytes'
    elif value.HasField('float_list'):
      type_string = 'float'
    elif value.HasField('int64_list'):
      type_string = 'int64'
    tf.logging.info('%s: [%s]', name, type_string)
  tf.logging.info('====')


def _PrintFiles():
  entry = 0
  for filepath in tf.gfile.Glob(FLAGS.input_filepattern):
    records = tf.compat.v1.io.tf_record_iterator(filepath)
    for serialized in records:
      if entry < FLAGS.skip_first_n:
        entry += 1
        continue
      if FLAGS.print_only_n >= 0 and (entry - FLAGS.skip_first_n >
                                      FLAGS.print_only_n):
        break
      if FLAGS.count_only:
        entry += 1
        if (entry % 100000) == 0:
          tf.logging.info('Counted %d entries so far...', entry)
        continue
      assert FLAGS.input_format == 'tf.Example'
      ex = tf.train.Example()
      ex.ParseFromString(serialized)
      if entry == FLAGS.skip_first_n:
        _PrintHeader(ex)
      text_format = _CustomShortDebugString(ex) if FLAGS.abbreviated else str(
          ex)
      tf.logging.info('== Record [%d]\n%s', entry, text_format)
      entry += 1
  tf.logging.info('== Total entries: %d', entry)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  _PrintFiles()


if __name__ == '__main__':
  tf.app.run(main)
