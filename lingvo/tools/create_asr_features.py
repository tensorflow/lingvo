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
"""Encode the audio tarball contents into tfrecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import random
import re

import tarfile

import tensorflow as tf

from lingvo.tools import audio_lib

tf.flags.DEFINE_string('input_tarball', '', 'Input .tar.gz file.')
tf.flags.DEFINE_string('input_text', '', 'Reference text.')
tf.flags.DEFINE_string('output_template', '', 'File of tfrecords.')

tf.flags.DEFINE_bool('dump_transcripts', False,
                     'First pass through the tarball to read the transcripts.')
tf.flags.DEFINE_string('transcripts_filepath', '',
                       'Where to put the transcripts.')
tf.flags.DEFINE_bool('generate_tfrecords', False,
                     'Second pass generates the tf records')

tf.flags.DEFINE_integer('shard_id', -1, 'Processor shard.')
tf.flags.DEFINE_integer(
    'num_shards', -1,
    'Number of processor shards. Must divide num_output_shards.')
tf.flags.DEFINE_integer('output_range_begin', -1, 'Begin of output shard IDs.')
tf.flags.DEFINE_integer('output_range_end', -1, 'End of output shard IDs.')
tf.flags.DEFINE_integer('num_output_shards', -1,
                        'Total number of output shards.')

FLAGS = tf.flags.FLAGS


def _MakeBytesFeature(unicode_array):
  value = [tf.compat.as_bytes(w) for w in unicode_array]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _MakeInt64Feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _MakeFloatFeature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _MakeTfExample(uttid, frames, text):
  flat_frames = frames.flatten()
  feature = {
      'uttid': _MakeBytesFeature([uttid]),
      'transcript': _MakeBytesFeature([text.lower()]),
      'frames': _MakeFloatFeature(flat_frames)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _ReadTranscriptions():
  """Read all transcription files from the tarball.

  Returns:
    A map of utterance id to upper case transcription.
  """
  tar = tarfile.open(FLAGS.input_tarball, mode='r:gz')
  n = 0
  tf.logging.info('First pass: loading text files...')
  # TODO(drpng): there's more information in the following files:
  # LibriSpeech/LICENSE.TXT
  # LibriSpeech/README.TXT
  # LibriSpeech/CHAPTERS.TXT
  # LibriSpeech/SPEAKERS.TXT
  # LibriSpeech/BOOKS.TXT
  trans = {}
  for tarinfo in tar:
    if not tarinfo.isreg():
      continue
    n += 1
    if 0 == n % 10000:
      tf.logging.info('Scanned %d entries...', n)
    if not tarinfo.name.endswith('.trans.txt'):
      continue
    # The file LibriSpeech/dev-clean/3170/137482/3170-137482.trans.txt
    # will contain lines such as:
    # 3170-137482-0000 WITH AN EDUCATION WHICH OUGHT TO ...
    # 3170-137482-0001 I WAS COMPELLED BY POVERTY ...
    key = tarinfo.name.strip('.trans.txt')
    f = tar.extractfile(tarinfo)
    u = 0
    for l in f.readlines():
      uttid, txt = l.strip('\n').split(' ', 1)
      trans[uttid] = txt
      u += 1
    tf.logging.info('[%s] = %d utterances', key, u)
    f.close()
  return trans


def _DumpTranscripts():
  trans = _ReadTranscriptions()
  with open(FLAGS.transcripts_filepath, 'w') as f:
    for uttid in sorted(trans):
      f.write('%s %s\n' % (uttid, trans[uttid]))


def _LoadTranscriptionsFromFile():
  trans = {}
  with open(FLAGS.transcripts_filepath, 'r') as f:
    for line in f.readlines():
      uttid, txt = line.strip('\n').split(' ', 1)
      trans[uttid] = txt
  return trans


def _MakeLogMelFromTensorflowBuiltin(tf_wav_bytes):
  sample_rate, audio = audio_lib.DecodeWav(tf_wav_bytes)
  static_sample_rate = 16000
  with tf.control_dependencies(
      [tf.assert_equal(sample_rate, static_sample_rate)]):
    log_mel = audio_lib.AudioToMfcc(static_sample_rate, audio, 25, 25, 40)
  return log_mel


def _OpenSubShards():
  tf.logging.info('Shards: %d to %d', FLAGS.output_range_begin,
                  FLAGS.output_range_end)
  recordio_writers = []
  for s in range(FLAGS.output_range_begin, FLAGS.output_range_end):
    filepath = FLAGS.output_template % (s, FLAGS.num_output_shards)
    tf.logging.info('Opening output shard: %s', filepath)
    recordio_writers += [tf.python_io.TFRecordWriter(filepath)]
  return recordio_writers


def _CloseSubShards(files):
  for f in files:
    f.close()


def _SelectRandomShard(files):
  subshard = random.randint(0, len(files) - 1)
  return files[subshard]


def _CreateAsrFeatures():
  # First pass: extract transcription files.
  if os.path.exists(FLAGS.transcripts_filepath):
    trans = _LoadTranscriptionsFromFile()
  else:
    tf.logging.info('Running first pass on the fly')
    trans = _ReadTranscriptions()
  tf.logging.info('Total transcripts: %d', len(trans))
  tf_bytes = tf.placeholder(dtype=tf.string)
  log_mel = audio_lib.ExtractLogMelFeatures(tf_bytes)
  # Second pass: transcode the flac.
  tar = tarfile.open(FLAGS.input_tarball, mode='r:gz')
  n = 0
  recordio_writers = _OpenSubShards()
  tfconf = tf.ConfigProto()
  tfconf.gpu_options.allow_growth = True
  with tf.Session(config=tfconf) as sess:
    for tarinfo in tar:
      if not tarinfo.name.endswith('.flac'):
        continue
      n += 1
      if n % FLAGS.num_shards != FLAGS.shard_id:
        continue
      uttid = re.sub('.*/(.+)\\.flac', '\\1', tarinfo.name)
      f = tar.extractfile(tarinfo)
      wav_bytes = audio_lib.DecodeFlacToWav(f.read())
      f.close()
      frames = sess.run(log_mel, feed_dict={tf_bytes: wav_bytes})
      assert uttid in trans, uttid
      num_words = len(trans[uttid])
      tf.logging.info('utt[%d]: %s [%d frames, %d words]', n, uttid,
                      frames.shape[1], num_words)
      ex = _MakeTfExample(uttid, frames, trans[uttid])
      outf = _SelectRandomShard(recordio_writers)
      outf.write(ex.SerializeToString())
    tar.close()
  _CloseSubShards(recordio_writers)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dump_transcripts:
    _DumpTranscripts()
  elif FLAGS.generate_tfrecords:
    _CreateAsrFeatures()
  else:
    tf.logging.error(
        'Nothing to do! Use --dump_transcripts or --generate_tfrecords')


if __name__ == '__main__':
  tf.app.run(main)
