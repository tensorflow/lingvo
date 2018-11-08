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
"""Tests for asr_frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf

from lingvo.core import asr_frontend
from lingvo.core import py_utils
from lingvo.core import test_helper
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


class AsrFrontendTest(tf.test.TestCase):

  def _GetPcm(self):
    with open(
        test_helper.test_src_dir_path('tools/testdata/gan_or_vae.wav'),
        'r') as f:
      wavdata = f.read()
      result = contrib_audio.decode_wav(wavdata)
      # Remove the last dimension: channel is 1.
      audio = py_utils.HasShape(result.audio, [75900, 1])
      audio = tf.squeeze(audio, axis=1)
      # Returns audio as batch-major data with a single batch.
      return result.sample_rate, tf.expand_dims(audio, axis=0)

  def _CreateFrontendParams(self):
    p = asr_frontend.MelFrontend.Params()
    p.sample_rate = 24000.
    p.num_bins = 2
    p.noise_scale = 0.
    self.params = p
    self._ref_mean = (13.38236332, 13.2698698, 13.45229626, 13.26469517,
                      13.46731281, 13.31649303)
    self._ref_stddev = (1.52104115, 1.27433181, 1.41266346, 1.27072334,
                        1.41251481, 1.28583682)

  def testMelFeatures(self):
    self._CreateFrontendParams()
    p = self.params
    mel_frontend = p.cls(p)
    sample_rate, pcm = self._GetPcm()
    pcm *= 32768
    log_mel, paddings = mel_frontend.FPropDefaultTheta(pcm)
    with self.session() as sess:
      pcm = sess.run(pcm)
      tf.logging.info('pcm: ~ %s = %s', pcm.shape, pcm)
      self.assertGreater(33000, np.amax(pcm))
      self.assertGreater(np.amax(pcm), 2.)
      log_mel, paddings, sample_rate = sess.run(
          [log_mel, paddings, sample_rate])
      self.assertEqual(sample_rate, p.sample_rate)
      self.assertAllEqual(paddings, np.zeros_like(paddings))
      # log_mel ~ [batch, time, feature_size, channel]
      tf.logging.info('mel ~ %s', log_mel.shape)
      # Squeeze the batch and channel dimensions out.
      log_mel = np.squeeze(log_mel, axis=(0, 3))
      t = log_mel.shape[0]
      mu = np.sum(log_mel, axis=0) / t
      d = log_mel - mu
      v = np.sum(d * d, axis=0) / (t - 1)
      s = np.sqrt(v)
      tf.logging.info('Found mean = %s', mu)
      tf.logging.info('Found stddev = %s', s)
      self.assertAllClose(mu, self._ref_mean, atol=1e-4)
      self.assertAllClose(s, self._ref_stddev, atol=1e-3)

  def testMelMeanVarNormalization(self):
    self._CreateFrontendParams()
    p = self.params
    p.per_bin_mean = self._ref_mean[:p.num_bins]
    p.per_bin_stddev = self._ref_stddev[:p.num_bins]
    mel_frontend = p.cls(p)
    _, pcm = self._GetPcm()
    pcm *= 32768
    log_mel, _ = mel_frontend.FPropDefaultTheta(pcm)
    with self.session() as sess:
      log_mel = sess.run(log_mel)
      # log_mel ~ [batch, time, feature_size, channel]
      tf.logging.info('mel ~ %s', log_mel.shape)
      # Squeeze the batch and channel dimensions out.
      log_mel = np.squeeze(log_mel, axis=(0, 3))
      t = log_mel.shape[0]
      mu = np.sum(log_mel, axis=0) / t
      d = log_mel - mu
      v = np.sum(d * d, axis=0) / (t - 1)
      s = np.sqrt(v)
      # Only take the base bin values:
      mu = mu[:p.num_bins]
      s = s[:p.num_bins]
      self.assertAllClose(mu, np.zeros_like(mu), atol=1e-4)
      self.assertAllClose(s, np.ones_like(s), atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
