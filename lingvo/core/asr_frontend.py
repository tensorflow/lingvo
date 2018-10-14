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
"""Layers to construct an ASR frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import py_utils


def _NextPowerOfTwo(i):
  return math.pow(2, math.ceil(math.log(i, 2)))


class MelFrontend(base_layer.BaseLayer):
  """A layer that implements mel feature extraction.

  This is expressed in pure TensorFlow and without reference to external
  resources.

  The frontend implements the following stages:
  `Framer -> Window -> FFT -> FilterBank -> MeanStdDev -> FrameStack`
  ` -> SubSample`
  """

  @classmethod
  def Params(cls):
    p = super(MelFrontend, cls).Params()
    p.name = 'frontend'
    p.Define('sample_rate', 16000.0, 'Sample rate in Hz')
    p.Define('frame_size_ms', 25.0,
             'Amount of data grabbed for each frame during analysis')
    p.Define('frame_step_ms', 10.0, 'Number of ms to jump between frames')
    p.Define('num_bins', 80, 'Number of bins in the mel-spectrogram output')
    p.Define('lower_edge_hertz', 125.0,
             'The lowest frequency of the mel-spectrogram analsis')
    p.Define('upper_edge_hertz', 7600.0,
             'The highest frequency of the mel-spectrogram analsis')
    p.Define('preemph', 0.97,
             'The first-order filter coefficient used for preemphasis')
    p.Define('noise_scale', 8.0,
             'The amount of noise (in 16-bit LSB units) to add')
    p.Define(
        'window_fn', 'HANNING',
        'Window function to apply (valid values are "HANNING", '
        'and None)')
    p.Define(
        'pad_end', False,
        'Whether to pad the end of `signals` with zeros when the provided '
        'frame length and step produces a frame that lies partially past '
        'its end.')
    p.Define(
        'per_bin_mean', None,
        'Per-bin (num_bins) means for normalizing the spectrograms. '
        'Defaults to zeros.')
    p.Define('per_bin_stddev', None, 'Per-bin (num_bins) standard deviations. '
             'Defaults to ones.')
    p.Define('left_context', 2, 'Number of left context frames to stack.')
    p.Define(
        'output_stride', 3,
        'Subsamples output frames by this factor (each output_stride '
        'frame is valid).')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MelFrontend, self).__init__(params)
    p = self.params
    # Make sure key params are in floating point.
    p.sample_rate = float(p.sample_rate)
    p.frame_step_ms = float(p.frame_step_ms)
    p.frame_size_ms = float(p.frame_size_ms)
    p.lower_edge_hertz = float(p.lower_edge_hertz)
    p.upper_edge_hertz = float(p.upper_edge_hertz)

    self._frame_step = int(round(p.sample_rate * p.frame_step_ms / 1000.0))
    self._frame_size = (int(round(p.sample_rate * p.frame_size_ms / 1000.0)) + 1
                       )  # +1 for the preemph
    # Overdrive means double FFT size.
    # Note: 2* because of overdrive
    self._fft_size = 2 * int(max(512, _NextPowerOfTwo(self._frame_size)))

    self._CreateWindowFunction()

    # Mean/stddev.
    if p.per_bin_mean is None:
      p.per_bin_mean = [0.0] * p.num_bins
    if p.per_bin_stddev is None:
      p.per_bin_stddev = [1.0] * p.num_bins
    assert len(p.per_bin_mean) == p.num_bins
    assert len(p.per_bin_stddev) == p.num_bins

  def _CreateWindowFunction(self):
    p = self.params
    if p.window_fn is None:
      self._window_fn = None
    elif p.window_fn == 'HANNING':

      def _HanningWindow(frame_size, dtype):
        return tf.contrib.signal.hann_window(frame_size, dtype=dtype)

      self._window_fn = _HanningWindow
    else:
      raise ValueError('Illegal value %r for window_fn param' % (p.window_fn,))

  @property
  def window_frame_size(self):
    return self._frame_size

  @property
  def window_frame_step(self):
    return self._frame_step

  def FProp(self, theta, pcm_audio_data):
    """Perform signal processing on a sequence of PCM data.

    Args:
      theta: Layer theta.
      pcm_audio_data: int16 or float32 tensor of PCM audio data, scaled to +/-
        32768 (versus 0..1!). Shaped: [batch, frame_count]

    Returns:
      src_inputs shaped [batch, time, features, 1], src_paddings
    """
    p = self.params
    batch_size = py_utils.GetShape(pcm_audio_data, 2)[0]
    mel_spectrogram_norm = self._FPropChunk(theta, pcm_audio_data)

    # Stacking across the whole sequence.
    assert p.left_context == 2, 'Only p.left context 2 is implemented.'
    first_frame = mel_spectrogram_norm[:, 0:1, :]
    padded_mel_spectrogram = tf.concat(
        (first_frame, first_frame, mel_spectrogram_norm), axis=1)
    frame_count = tf.shape(padded_mel_spectrogram)[1] // 3
    triple_mel = tf.reshape(padded_mel_spectrogram[:, 0:3 * frame_count, :],
                            [batch_size, frame_count, 3 * p.num_bins])
    src_paddings = 0 * tf.reduce_sum(triple_mel, axis=2)

    # Add feature dim. Shape = [batch, time, features, 1]
    src_inputs = triple_mel
    src_inputs = tf.expand_dims(triple_mel, -1)
    return src_inputs, src_paddings

  def _ApplyPreemphasis(self, framed_signal):
    p = self.params
    preemphasized = (
        framed_signal[:, :, 1:] - p.preemph * framed_signal[:, :, 0:-1])
    return preemphasized

  def _FPropChunk(self, theta, pcm_audio_chunk):
    p = self.params
    pcm_audio_chunk = tf.cast(pcm_audio_chunk, tf.float32)
    framed_signal = tf.contrib.signal.frame(pcm_audio_chunk, self._frame_size,
                                            self._frame_step, p.pad_end)
    # Pre-emphasis.
    if p.preemph != 1.0:
      preemphasized = self._ApplyPreemphasis(framed_signal)
    else:
      preemphasized = framed_signal[:-1]

    # Noise.
    if p.noise_scale > 0.0:
      noise_signal = tf.random_normal(
          tf.shape(preemphasized), stddev=p.noise_scale, mean=0.0)
    else:
      noise_signal = 0.0

    # Apply window fn.
    windowed_signal = preemphasized + noise_signal
    if self._window_fn is not None:
      window = self._window_fn(self._frame_size - 1, framed_signal.dtype)
      windowed_signal *= window

    mel_spectrogram = self._MelSpectrogram(windowed_signal)

    output_floor = 1.0
    mel_spectrogram_log = tf.log(
        tf.maximum(float(output_floor), mel_spectrogram))

    # Mean and stddev.
    mel_spectrogram_norm = (
        (mel_spectrogram_log - tf.convert_to_tensor(p.per_bin_mean)) /
        tf.convert_to_tensor(p.per_bin_stddev))
    return mel_spectrogram_norm

  def _MelSpectrogram(self, signal):
    """Computes the mel spectrogram from a waveform signal.

    Args:
      signal: f32 Tensor, shaped [batch_size, num_samples]

    Returns:
      features: f32 Tensor, shaped [batch_size, num_frames, mel_channels]
    """
    p = self.params
    # FFT.
    real_frequency_spectrogram = tf.spectral.rfft(signal, [self._fft_size])
    magnitude_spectrogram = tf.abs(real_frequency_spectrogram)

    # Shape of magnitude_spectrogram is num_frames x (fft_size/2+1)
    # Mel_weight is [num_spectrogram_bins, num_mel_bins]
    mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=p.num_bins,
        num_spectrogram_bins=self._fft_size // 2 + 1,
        sample_rate=p.sample_rate,
        lower_edge_hertz=p.lower_edge_hertz,
        upper_edge_hertz=p.upper_edge_hertz,
        dtype=tf.float32)
    # Weight matrix implemented in the magnitude domain.
    batch_size, num_frames, fft_channels = py_utils.GetShape(
        magnitude_spectrogram, 3)
    mel_spectrogram = tf.matmul(
        tf.reshape(magnitude_spectrogram,
                   [batch_size * num_frames, fft_channels]), mel_weight_matrix)
    mel_spectrogram = tf.reshape(mel_spectrogram,
                                 [batch_size, num_frames, p.num_bins])

    return mel_spectrogram
