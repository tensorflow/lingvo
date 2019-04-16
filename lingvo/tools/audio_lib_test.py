# -*- coding: utf-8 -*-
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
"""Tests for audio_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tools import audio_lib

# The testdata contains: (soxi .../gan_or_vae.wav)
# Channels       : 1
# Sample Rate    : 24000
# Precision      : 16-bit
# Duration       : 00:00:03.16 = 75900 samples ~ 237.188 CDDA sectors


class AudioLibTest(test_utils.TestCase):

  def testDecodeFlacToWav(self):
    with open(
        test_helper.test_src_dir_path('tools/testdata/gan_or_vae.wav'),
        'r') as f:
      wav = f.read()
    with open(
        test_helper.test_src_dir_path('tools/testdata/gan_or_vae.flac'),
        'r') as f:
      flac = f.read()
    tf.logging.info('flac: %d bytes', len(flac))
    try:
      converted = audio_lib.DecodeFlacToWav(flac)
      tf.logging.info('wav: %d bytes, converted: %d bytes', len(wav),
                      len(converted))
      self.assertEqual(wav, converted)
    except OSError:
      # sox is not installed, ignore this test.
      pass

  def testDecodeWav(self):
    with open(
        test_helper.test_src_dir_path('tools/testdata/gan_or_vae.wav'),
        'r') as f:
      wav = f.read()
    with self.session() as sess:
      sample_rate, audio = sess.run(audio_lib.DecodeWav(wav))
      self.assertEqual(24000, sample_rate)
      self.assertEqual(75900, len(audio))

  def testAudioToMfcc(self):
    with open(
        test_helper.test_src_dir_path('tools/testdata/gan_or_vae.wav'),
        'r') as f:
      wav = f.read()
    sample_rate, audio = audio_lib.DecodeWav(wav)
    static_sample_rate = 24000
    mfcc = audio_lib.AudioToMfcc(static_sample_rate, audio, 32, 25, 40)
    with self.session() as sess:
      audio_sample_rate, mfcc = sess.run([sample_rate, mfcc])
      assert audio_sample_rate == static_sample_rate
      self.assertAllEqual(mfcc.shape, [1, 126, 40])

  def testExtractLogMelFeatures(self):
    with open(
        test_helper.test_src_dir_path('tools/testdata/gan_or_vae.16k.wav'),
        'r') as f:
      wav = f.read()

    wav_bytes_t = tf.constant(wav, dtype=tf.string)
    log_mel_t = audio_lib.ExtractLogMelFeatures(wav_bytes_t)

    with self.session() as sess:
      log_mel = sess.run(log_mel_t)
      # Expect 314, 80 dimensional channels.
      self.assertAllEqual(log_mel.shape, [1, 314, 80, 1])


if __name__ == '__main__':
  tf.test.main()
