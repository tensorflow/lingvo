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
"""Tests for asr frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from lingvo.core import py_utils
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.asr import frontend


class AsrFrontendTest(test_utils.TestCase):

  def _GetPcm(self):
    """Gets sample wav file pcm samples.

    Returns:
      (sample_rate, mono_audio) where mono_audio is of shape
      [batch (=1), samples].
    """
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
    p = frontend.MelAsrFrontend.Params()
    p.sample_rate = 24000.
    p.num_bins = 2
    p.noise_scale = 0.
    self.params = p

  def testNullAsrFrontendConfig(self):
    p = frontend.NullAsrFrontend.Params()
    p.name = 'null'
    fe = p.cls(p)
    config = fe.config
    self.assertTrue(config.is_null)
    self.assertEqual(config.src_type, 'none')
    self.assertEqual(config.output_dim, -1)
    self.assertEqual(config.input_frame_ratio, 1.0)

  def testMelFeaturesUnstackedConfig(self):
    self._CreateFrontendParams()
    p = self.params
    fe = p.cls(p)
    config = fe.config
    self.assertFalse(config.is_null)
    self.assertEqual(config.src_type, 'pcm')
    self.assertEqual(config.src_pcm_scale, 32768.0)
    self.assertEqual(config.src_pcm_sample_rate, 16000.0)
    self.assertEqual(config.output_dim, 2)
    # Approx 34 output frames per second.
    self.assertEqual(config.input_frame_ratio, 480.0)

  def testMelFeaturesLeftStackedConfig(self):
    self._CreateFrontendParams()
    p = self.params
    p.stack_left_context = 2
    fe = p.cls(p)
    config = fe.config
    self.assertFalse(config.is_null)
    self.assertEqual(config.src_type, 'pcm')
    self.assertEqual(config.src_pcm_scale, 32768.0)
    self.assertEqual(config.src_pcm_sample_rate, 16000.0)
    self.assertEqual(config.output_dim, 6)
    # Approx 12 output frames per second.
    self.assertEqual(config.input_frame_ratio, 1440.0)

  def testMelFeaturesUnstacked(self):
    self._CreateFrontendParams()
    p = self.params
    mel_frontend = p.cls(p)
    sample_rate, pcm = self._GetPcm()
    pcm *= 32768

    # Convert to 4D [batch, time, packet, channels].
    sample_count = tf.shape(pcm)[1]
    packet_size = 11  # A non-round number.
    trimmed_pcm = pcm[:, 0:(sample_count // packet_size) * packet_size]
    src_inputs = tf.reshape(trimmed_pcm, (1, -1, packet_size, 1))
    paddings = tf.zeros(tf.shape(src_inputs)[0:2])

    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=src_inputs, paddings=paddings))
    log_mel = outputs.src_inputs
    paddings = outputs.paddings
    with self.session() as sess:
      pcm = sess.run(pcm)
      tf.logging.info('pcm: ~ %s = %s', pcm.shape, pcm)
      self.assertGreater(33000, np.amax(pcm))
      self.assertGreater(np.amax(pcm), 2.)
      log_mel, paddings, sample_rate = sess.run(
          [log_mel, paddings, sample_rate])
      self.assertEqual(sample_rate, p.sample_rate)
      self.assertEqual(paddings.shape, log_mel.shape[0:2])
      self.assertAllEqual(paddings, np.zeros_like(paddings))
      # log_mel ~ [batch, time, feature_size, channel]
      tf.logging.info('mel ~ %s', log_mel.shape)
      self.assertEqual(log_mel.shape[2], 2)  # 2 bins
      # Squeeze the batch and channel dimensions out.
      log_mel = np.squeeze(log_mel, axis=(0, 3))
      t = log_mel.shape[0]
      mu = np.sum(log_mel, axis=0) / t
      d = log_mel - mu
      v = np.sum(d * d, axis=0) / (t - 1)
      s = np.sqrt(v)
      tf.logging.info('Found mean = %s', mu)
      tf.logging.info('Found stddev = %s', s)
      ref_unstacked_mean = [13.46184731, 13.30099297]
      ref_unstacked_stddev = [1.3840059, 1.24434352]
      self.assertAllClose(mu, ref_unstacked_mean, atol=1e-4)
      self.assertAllClose(s, ref_unstacked_stddev, atol=1e-3)

  def testMelFeaturesLeftStacked(self):
    self._CreateFrontendParams()
    p = self.params
    p.stack_left_context = 2
    mel_frontend = p.cls(p)
    sample_rate, pcm = self._GetPcm()
    pcm *= 32768

    # Convert to 4D [batch, time, packet, channels].
    sample_count = tf.shape(pcm)[1]
    packet_size = 11  # A non-round number.
    trimmed_pcm = pcm[:, 0:(sample_count // packet_size) * packet_size]
    src_inputs = tf.reshape(trimmed_pcm, (1, -1, packet_size, 1))
    paddings = tf.zeros(tf.shape(src_inputs)[0:2])

    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=src_inputs, paddings=paddings))
    log_mel = outputs.src_inputs
    paddings = outputs.paddings
    with self.session() as sess:
      pcm = sess.run(pcm)
      tf.logging.info('pcm: ~ %s = %s', pcm.shape, pcm)
      self.assertGreater(33000, np.amax(pcm))
      self.assertGreater(np.amax(pcm), 2.)
      log_mel, paddings, sample_rate = sess.run(
          [log_mel, paddings, sample_rate])
      self.assertEqual(sample_rate, p.sample_rate)
      self.assertEqual(paddings.shape, log_mel.shape[0:2])
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
      ref_mean = (13.38236332, 13.2698698, 13.45229626, 13.26469517,
                  13.46731281, 13.31649303)
      ref_stddev = (1.52104115, 1.27433181, 1.41266346, 1.27072334, 1.41251481,
                    1.28583682)
      self.assertAllClose(mu, ref_mean, atol=1e-4)
      self.assertAllClose(s, ref_stddev, atol=1e-3)

  def testMelFeaturesPaddedLeftStacked(self):
    self._CreateFrontendParams()
    p = self.params
    p.stack_left_context = 2
    mel_frontend = p.cls(p)
    sample_rate, pcm = self._GetPcm()
    pcm *= 32768

    # Convert to 4D [batch, time, packet, channels].
    sample_count = tf.shape(pcm)[1]
    packet_size = 11  # A non-round number.
    trimmed_pcm = pcm[:, 0:(sample_count // packet_size) * packet_size]
    src_inputs = tf.reshape(trimmed_pcm, (1, -1, packet_size, 1))

    # Create paddings such that the first 455 packets are unpadded.
    paddings = tf.concat([
        tf.zeros([1, 455], dtype=tf.float32),
        tf.ones([1, tf.shape(src_inputs)[1] - 455], dtype=tf.float32)
    ],
                         axis=1)
    # frame_step=240, frame_size=601, +1202 left padded frames
    # 455 packets * 11 frames rounds = 5005 frames, rounds down to 21 mel
    # frames. Divide by 3 for stacking = 7.
    expected_unpadded = 7

    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=src_inputs, paddings=paddings))
    log_mel = outputs.src_inputs
    paddings = outputs.paddings
    with self.session() as sess:
      pcm = sess.run(pcm)
      tf.logging.info('pcm: ~ %s = %s', pcm.shape, pcm)
      self.assertGreater(33000, np.amax(pcm))
      self.assertGreater(np.amax(pcm), 2.)
      log_mel, paddings, sample_rate = sess.run(
          [log_mel, paddings, sample_rate])
      self.assertEqual(sample_rate, p.sample_rate)
      self.assertEqual(paddings.shape, log_mel.shape[0:2])
      self.assertAllEqual(paddings[:, 0:expected_unpadded],
                          np.zeros([1, expected_unpadded]))
      self.assertAllEqual(paddings[:, expected_unpadded:],
                          np.ones([1, paddings.shape[1] - expected_unpadded]))
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
      ref_mean = (13.38236332, 13.2698698, 13.45229626, 13.26469517,
                  13.46731281, 13.31649303)
      ref_stddev = (1.52104115, 1.27433181, 1.41266346, 1.27072334, 1.41251481,
                    1.28583682)
      self.assertAllClose(mu, ref_mean, atol=1e-4)
      self.assertAllClose(s, ref_stddev, atol=1e-3)

  def testMelMeanVarNormalization(self):
    self._CreateFrontendParams()
    p = self.params
    p.stack_left_context = 2
    ref_mean = (13.38236332, 13.2698698, 13.45229626, 13.26469517, 13.46731281,
                13.31649303)
    ref_stddev = (1.52104115, 1.27433181, 1.41266346, 1.27072334, 1.41251481,
                  1.28583682)
    p.per_bin_mean = ref_mean[:p.num_bins]
    p.per_bin_stddev = ref_stddev[:p.num_bins]
    mel_frontend = p.cls(p)
    _, pcm = self._GetPcm()
    pcm *= 32768

    # Convert to 4D [batch, time, packet, channels].
    sample_count = tf.shape(pcm)[1]
    packet_size = 11  # A non-round number.
    trimmed_pcm = pcm[:, 0:(sample_count // packet_size) * packet_size]
    src_inputs = tf.reshape(trimmed_pcm, (1, -1, packet_size, 1))
    paddings = tf.zeros(tf.shape(src_inputs)[0:2])

    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=src_inputs, paddings=paddings))
    log_mel = outputs.src_inputs
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

  def testMelFeaturesUnstacked2D(self):
    # TODO(laurenzo): Remove this test once 2D inputs support removed.
    self._CreateFrontendParams()
    p = self.params
    mel_frontend = p.cls(p)
    sample_rate, pcm = self._GetPcm()
    pcm *= 32768

    # Leave in 2D [batch, time].
    src_inputs = pcm
    paddings = tf.zeros(tf.shape(src_inputs)[0:2])

    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=src_inputs, paddings=paddings))
    log_mel = outputs.src_inputs
    paddings = outputs.paddings
    with self.session() as sess:
      pcm = sess.run(pcm)
      tf.logging.info('pcm: ~ %s = %s', pcm.shape, pcm)
      self.assertGreater(33000, np.amax(pcm))
      self.assertGreater(np.amax(pcm), 2.)
      log_mel, paddings, sample_rate = sess.run(
          [log_mel, paddings, sample_rate])
      self.assertEqual(sample_rate, p.sample_rate)
      self.assertEqual(paddings.shape, log_mel.shape[0:2])
      self.assertAllEqual(paddings, np.zeros_like(paddings))
      # log_mel ~ [batch, time, feature_size, channel]
      tf.logging.info('mel ~ %s', log_mel.shape)
      self.assertEqual(log_mel.shape[2], 2)  # 2 bins
      # Squeeze the batch and channel dimensions out.
      log_mel = np.squeeze(log_mel, axis=(0, 3))
      t = log_mel.shape[0]
      mu = np.sum(log_mel, axis=0) / t
      d = log_mel - mu
      v = np.sum(d * d, axis=0) / (t - 1)
      s = np.sqrt(v)
      tf.logging.info('Found mean = %s', mu)
      tf.logging.info('Found stddev = %s', s)
      ref_unstacked_mean = [13.46184731, 13.30099297]
      ref_unstacked_stddev = [1.3840059, 1.24434352]
      self.assertAllClose(mu, ref_unstacked_mean, atol=1e-4)
      self.assertAllClose(s, ref_unstacked_stddev, atol=1e-3)

  def testMelFeaturesUnstacked3D(self):
    # TODO(laurenzo): Remove this test once 3D inputs support removed.
    self._CreateFrontendParams()
    p = self.params
    mel_frontend = p.cls(p)
    sample_rate, pcm = self._GetPcm()
    pcm *= 32768

    # Leave in 3D [batch, time, 1].
    src_inputs = tf.expand_dims(pcm, axis=2)
    paddings = tf.zeros(tf.shape(src_inputs)[0:2])

    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=src_inputs, paddings=paddings))
    log_mel = outputs.src_inputs
    paddings = outputs.paddings
    with self.session() as sess:
      pcm = sess.run(pcm)
      tf.logging.info('pcm: ~ %s = %s', pcm.shape, pcm)
      self.assertGreater(33000, np.amax(pcm))
      self.assertGreater(np.amax(pcm), 2.)
      log_mel, paddings, sample_rate = sess.run(
          [log_mel, paddings, sample_rate])
      self.assertEqual(sample_rate, p.sample_rate)
      self.assertEqual(paddings.shape, log_mel.shape[0:2])
      self.assertAllEqual(paddings, np.zeros_like(paddings))
      # log_mel ~ [batch, time, feature_size, channel]
      tf.logging.info('mel ~ %s', log_mel.shape)
      self.assertEqual(log_mel.shape[2], 2)  # 2 bins
      # Squeeze the batch and channel dimensions out.
      log_mel = np.squeeze(log_mel, axis=(0, 3))
      t = log_mel.shape[0]
      mu = np.sum(log_mel, axis=0) / t
      d = log_mel - mu
      v = np.sum(d * d, axis=0) / (t - 1)
      s = np.sqrt(v)
      tf.logging.info('Found mean = %s', mu)
      tf.logging.info('Found stddev = %s', s)
      ref_unstacked_mean = [13.46184731, 13.30099297]
      ref_unstacked_stddev = [1.3840059, 1.24434352]
      self.assertAllClose(mu, ref_unstacked_mean, atol=1e-4)
      self.assertAllClose(s, ref_unstacked_stddev, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
