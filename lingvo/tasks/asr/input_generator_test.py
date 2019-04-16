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
"""Tests for asr.input_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from six.moves import range

import tensorflow as tf
from lingvo.core import test_utils

from lingvo.tasks.asr import input_generator


def _MakeBytesFeature(unicode_array):
  value = [tf.compat.as_bytes(w) for w in unicode_array]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _MakeInt64Feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _MakeFloatFeature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


FRAME_SIZE = 40


def _MakeFrames(utt_len):
  return np.random.normal(size=FRAME_SIZE * utt_len)


def _MakeTfExample(uttid, frames, text):
  flat_frames = frames.flatten()
  feature = {
      'uttid': _MakeBytesFeature([uttid]),
      'transcript': _MakeBytesFeature([text]),
      'frames': _MakeFloatFeature(flat_frames)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _LookupNested(nm, nested_key):
  key_array = nested_key.split('.')
  val = nm
  for k in key_array:
    val = val[k]
  return val


class InputTest(test_utils.TestCase):

  def _GenerateExamples(self, output_filepath):
    example_def = [('utt1', (1234, 'HELLO WORLD')),
                   ('utt2', (568, 'TIRED WITH ALL THESE')),
                   ('utt3', (778, 'WOULD THAT IT WERE SO EASY'))]
    self._example_def = dict(example_def)
    tf_examples = []
    for xdef in example_def:
      tf_examples.append(
          _MakeTfExample(xdef[0], _MakeFrames(xdef[1][0]), xdef[1][1]))
    with tf.python_io.TFRecordWriter(output_filepath) as outf:
      for ex in tf_examples:
        outf.write(ex.SerializeToString())

  def _GenerateSetup(self, append_eos_frame):
    tfrecords_filepath = os.path.join(tf.test.get_temp_dir(),
                                      'simple.tfrecords')
    self._GenerateExamples(tfrecords_filepath)
    p = input_generator.AsrInput.Params()
    p.file_pattern = 'tfrecord:' + tfrecords_filepath
    p.frame_size = FRAME_SIZE
    p.target_max_length = 30
    p.bucket_upper_bound = [2560]
    p.bucket_batch_limit = [3]
    p.append_eos_frame = append_eos_frame
    return p

  def _AssertAllOnes(self, np_data):
    self.assertAllEqual(np_data, np.ones_like(np_data))

  def _AssertAllZeros(self, np_data):
    self.assertAllEqual(np_data, np.zeros_like(np_data))

  def _AssertShapesAsExpected(self, shapes, expected):
    for key, expected_shape in sorted(expected.items()):
      self.assertAllEqual(
          _LookupNested(shapes, key), expected_shape, msg='Shape of %s' % key)

  def _TestAsrInput(self, params):
    p = params
    with self.session(use_gpu=False) as sess:
      inp = input_generator.AsrInput(p)
      batch = inp.GetPreprocessedInputBatch()
      vals = sess.run(batch)
      shapes = vals.Transform(lambda x: x.shape)
      tf.logging.info('Shapes: %s', shapes.DebugString())
      # sample_ids      (3, 1)
      # src.src_inputs  (3, 1235, 40, 1)
      # src.paddings    (3, 1235)
      # tgt.ids         (3, 30)
      # tgt.labels      (3, 30)
      # tgt.paddings    (3, 30)
      # tgt.weights     (3, 30)
      batch_size = p.bucket_batch_limit[0]
      max_num_frames = np.amax([xdef[0] for xdef in self._example_def.values()])
      if p.append_eos_frame:
        max_num_frames += 1
      tgt_shape = [batch_size, p.target_max_length]
      self._AssertShapesAsExpected(
          shapes, {
              'sample_ids': [batch_size, 1],
              'src.src_inputs': [batch_size, max_num_frames, p.frame_size, 1],
              'src.paddings': [batch_size, max_num_frames],
              'tgt.ids': tgt_shape,
              'tgt.labels': tgt_shape,
              'tgt.paddings': tgt_shape,
              'tgt.weights': tgt_shape
          })
      for b in range(batch_size):
        ex = vals.Transform(lambda x: x[b])
        ref = self._example_def[ex.sample_ids[0]]
        # Check source.
        ref_num_frames = ref[0]
        if p.append_eos_frame:
          ref_num_frames += 1
        ref_num_padding_frames = max_num_frames - ref_num_frames
        zero_frames = ex.src.src_inputs[max_num_frames -
                                        ref_num_padding_frames:]
        self._AssertAllZeros(zero_frames)
        zero_paddings = ex.src.paddings[:ref_num_frames]
        one_paddings = ex.src.paddings[ref_num_frames:]
        self._AssertAllZeros(zero_paddings)
        self._AssertAllOnes(one_paddings)
        # Check target. Something like this:
        # ids:      [1 a b c 2 2 2]
        # labels:   [a b c 2 2 2 2]
        # paddings: [0 0 0 1 1 1 1]
        ref_num_graphemes = len(ref[1])
        ref_tgt_ids_padding = p.target_max_length - ref_num_graphemes - 1
        self.assertEqual(ex.tgt.ids[0], p.tokenizer.target_sos_id)
        self.assertAllEqual(ex.tgt.ids[-ref_tgt_ids_padding:],
                            [2] * ref_tgt_ids_padding)
        self.assertAllEqual(ex.tgt.labels[-ref_tgt_ids_padding - 1:],
                            [2] * (ref_tgt_ids_padding + 1))
        self.assertAllEqual(ex.tgt.ids[1:1 + ref_num_graphemes],
                            ex.tgt.labels[:ref_num_graphemes])
        self._AssertAllZeros(ex.tgt.paddings[:ref_num_graphemes + 1])
        self._AssertAllOnes(ex.tgt.paddings[ref_num_graphemes + 1:])
        self._AssertAllOnes(ex.tgt.weights[:ref_num_graphemes + 1])
        self._AssertAllZeros(ex.tgt.weights[ref_num_graphemes + 1:])

  def testAsrInput(self):
    p = self._GenerateSetup(append_eos_frame=True)
    self._TestAsrInput(p)

  def testAsrInputWithoutEosFrame(self):
    p = self._GenerateSetup(append_eos_frame=False)
    self._TestAsrInput(p)


if __name__ == '__main__':
  tf.test.main()
