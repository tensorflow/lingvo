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
