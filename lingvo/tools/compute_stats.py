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
"""Compute stats from tfrecords files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

tf.flags.DEFINE_string('input_filepattern', '',
                       'File pattern of binary tfrecord files.')
tf.flags.DEFINE_integer('frame_size', 1, 'Size of the frame, for reshaping.')
tf.flags.DEFINE_integer('num_buckets', 8, 'Number of buckets for the length.')
tf.flags.DEFINE_string('feature_name', None, 'Name of feature to examine.')

FLAGS = tf.flags.FLAGS


class StatsCollector(object):

  def __init__(self,):
    self._num_examples = 0
    self._lengths = []
    self._num_frames = 0
    self._mean_acc = np.zeros(FLAGS.frame_size, dtype=np.float64)
    self._var_acc = np.zeros(FLAGS.frame_size, dtype=np.float64)

  def _AccumulateMoments(self, float_list):
    frames = np.reshape(float_list, [-1, FLAGS.frame_size])
    self._num_frames += frames.shape[0]
    self._mean_acc += np.sum(frames, axis=0)
    self._var_acc += np.sum(frames * frames, axis=0)

  def _ComputeMeanVar(self):
    mu = self._mean_acc / self._num_frames
    # The user is in charge of replacing NaNs with a floor value.
    v = np.sqrt(self._var_acc / self._num_frames - mu * mu)
    return mu, v

  def Accumulate(self, tf_ex):
    self._num_examples += 1
    if 0 == self._num_examples % 10000:
      tf.logging.info('Processing example %u...', self._num_examples)
    v = tf_ex.features.feature[FLAGS.feature_name]
    if v.HasField('float_list'):
      num_frames = len(v.float_list.value) // FLAGS.frame_size
      self._AccumulateMoments(v.float_list.value)
    elif v.HasField('int64_list'):
      num_frames = len(v.int64_list.value) // FLAGS.frame_size
    else:
      tf.logging.fatal(
          'Not sure what to do with value. '
          'Only float/int64 lists are supported: %s', v)
    self._lengths.append(num_frames)

  def _PrintLengthBuckets(self):
    sorted_lengths = sorted(self._lengths)
    num_buckets = FLAGS.num_buckets
    n = len(sorted_lengths)
    idx = (n * (np.array(range(num_buckets - 1)) + 1)) // num_buckets
    buckets = [sorted_lengths[i] for i in idx] + [sorted_lengths[-1]]
    tf.logging.info('== Buckets.')
    tf.logging.info('bucket upper limits: %s', buckets)
    tf.logging.info('Other candidates for last bucket:')
    tf.logging.info('  0.1%% loss: %u', sorted_lengths[int(n * .999)])
    tf.logging.info('    1%% loss: %u', sorted_lengths[int(n * .99)])
    tf.logging.info('    2%% loss: %u', sorted_lengths[int(n * .98)])

  def _PrintMeanVar(self):
    m, v = self._ComputeMeanVar()
    original = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    tf.logging.info('== Mean/variance.')
    tf.logging.info('mean = %s', m)
    tf.logging.info('var = %s', v)
    np.set_printoptions(**original)

  def Print(self):
    tf.logging.info('== Total number of examples: %u', self._num_examples)
    self._PrintLengthBuckets()
    self._PrintMeanVar()


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not FLAGS.feature_name:
    tf.logging.fatal('Use a --feature_name to specify what to bucketize on. '
                     'For instance, source_id for MT or frames for ASR.')
  stats = StatsCollector()
  for filepath in tf.gfile.Glob(FLAGS.input_filepattern):
    records = tf.compat.v1.io.tf_record_iterator(filepath)
    for serialized in records:
      ex = tf.train.Example()
      ex.ParseFromString(serialized)
      stats.Accumulate(ex)
  stats.Print()


if __name__ == '__main__':
  tf.app.run(main)
