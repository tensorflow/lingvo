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
"""Audio library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from lingvo.core import py_utils
from lingvo.tasks.asr import frontend as asr_frontend


# There are two ways to decode a wav in tensorflow:
# Through the tensorflow native audio decoder, exported
# via framework, or via tf.contrib.ffmpeg.decode_audio.
# While the latter could technically support FLAC, it does
# not. It also adds an extra dependency on ffmpeg.


def DecodeFlacToWav(input_bytes):
  """Decode a FLAC byte string to WAV."""
  p = subprocess.Popen(
      ['sox', '-t', 'flac', '-', '-t', 'wav', '-'],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  out, err = p.communicate(input=input_bytes)
  assert p.returncode == 0, err
  return out


def DecodeWav(input_bytes):
  """Decode a wav file from its contents.

  Args:
    input_bytes: a byte array or Tensor with the wav file contents.

  Returns:
    A pair of Tensor for sample rate, decoded samples.
  """
  result = contrib_audio.decode_wav(input_bytes)
  return result.sample_rate, result.audio


def AudioToMfcc(sample_rate, audio, window_size_ms, window_stride_ms,
                num_coefficients):
  window_size_samples = sample_rate * window_size_ms // 1000
  window_stride_samples = sample_rate * window_stride_ms // 1000
  spectrogram = contrib_audio.audio_spectrogram(
      audio,
      window_size=window_size_samples,
      stride=window_stride_samples,
      magnitude_squared=True)
  mfcc = contrib_audio.mfcc(
      spectrogram, sample_rate, dct_coefficient_count=num_coefficients)
  return mfcc


def ExtractLogMelFeatures(wav_bytes_t):
  """Create Log-Mel Filterbank Features from raw bytes.

  Args:
    wav_bytes_t: Tensor representing raw wav file as a string of bytes. It is
      currently assumed that the wav file is encoded at 16KHz (see DecodeWav,
      below).

  Returns:
    A Tensor representing three stacked log-Mel filterbank energies, sub-sampled
    every three frames.
  """

  def _CreateAsrFrontend():
    """Parameters corresponding to default ASR frontend."""
    p = asr_frontend.MelAsrFrontend.Params()
    p.sample_rate = 16000.
    p.frame_size_ms = 25.
    p.frame_step_ms = 10.
    p.num_bins = 80
    p.lower_edge_hertz = 125.
    p.upper_edge_hertz = 7600.
    p.preemph = 0.97
    p.noise_scale = 0.
    p.pad_end = False
    return p.cls(p)

  sample_rate, audio = DecodeWav(wav_bytes_t)
  audio *= 32768
  # Remove channel dimension, since we have a single channel.
  audio = tf.squeeze(audio, axis=1)
  # TODO(drpng): make batches.
  audio = tf.expand_dims(audio, axis=0)
  static_sample_rate = 16000
  mel_frontend = _CreateAsrFrontend()
  with tf.control_dependencies(
      [tf.assert_equal(sample_rate, static_sample_rate)]):
    outputs = mel_frontend.FPropDefaultTheta(
        py_utils.NestedMap(src_inputs=audio, paddings=tf.zeros_like(audio)))
    log_mel = outputs.src_inputs
  return log_mel
