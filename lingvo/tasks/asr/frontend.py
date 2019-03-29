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

import collections
import math

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import py_utils


# AsrFrontendConfig which defines characteristics of the frontend that may
# be relevant to interfacing code which needs to reason about inputs and
# outputs.
# Fields:
#   is_null: Whether this is the NullAsrFrontend.
#   src_type: Interpretation of the src_inputs. Can be one of 'none' or 'pcm'.
#   src_pcm_scale: If src_type is 'pcm', then this is the scale of each sample.
#     If normalized, this should be 1.0. If unnormalized from int16, then it
#     should be 32768.0.
#   src_pcm_sample_rate: Sample rate of the expected src PCM frames.
#   output_dim: Dimension of the output. Typically the number of mel bands
#     or equiv. May be -1 for unknown.
#   input_frame_ratio: Approximate ratio of the number of
#     input_frames / output_frames. Intended to be multiplied by output frames
#     (i.e. as part of bucket_bounds to arrive at input frames to the frontend).
AsrFrontendConfig = collections.namedtuple('AsrFrontendConfig', [
    'is_null',
    'src_type',
    'src_pcm_scale',
    'src_pcm_sample_rate',
    'output_dim',
    'input_frame_ratio',
])


def _NextPowerOfTwo(i):
  return math.pow(2, math.ceil(math.log(i, 2)))


class BaseAsrFrontend(base_layer.BaseLayer):
  """Base class for ASR frontends.

  An ASR frontend is responsible for performing feature extraction from the
  input in the cases where features are not precomputed as part of the
  dataset. In such cases, it would be typical for the input to consist of
  waveform data in some form.
  """

  @property
  def config(self):
    """Returns the AsrFrontendConfig namedtuple for this instance."""
    return self.GetConfigFromParams(self.params)

  @staticmethod
  def GetConfigFromParams(params):
    """Returns an AsrFrontendConfig namedtuple with vital config settings."""
    raise NotImplementedError()

  def FProp(self, theta, input_batch):
    """Generates ASR features for a batch.

    Shapes of the input_batch and output are dependent on the implementation
    and should be paired with the model's input format and encoder expectations.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      input_batch: A NestedMap with fields:

        - 'src_inputs' - The inputs tensor,
          compatible with model input. Expected to be of shape
          [batch, time, ...].
        - 'paddings' - The paddings tensor. It is expected to be of shape
          [batch, time].

    Returns:
      NestedMap of encoder inputs which can be passed directly to a
      compatible encoder and contains:

        - 'src_inputs': inputs to the encoder, minimally of shape
          [batch, time, ...].
        - 'paddings': a 0/1 tensor of shape [batch, time].
    """
    raise NotImplementedError()


class NullAsrFrontend(BaseAsrFrontend):
  """ASR frontend that just returns its input as FProp output."""

  @staticmethod
  def GetConfigFromParams(params):
    """Returns an AsrFrontendConfig namedtuple with vital config settings."""
    return AsrFrontendConfig(
        is_null=True,
        src_type='none',
        src_pcm_sample_rate=-1,
        src_pcm_scale=1.0,
        output_dim=-1,
        input_frame_ratio=1.0)

  def FProp(self, theta, input_batch):
    return input_batch.DeepCopy()


class MelAsrFrontend(BaseAsrFrontend):
  """An AsrFrontend that implements mel feature extraction from PCM frames.

  This is expressed in pure TensorFlow and without reference to external
  resources.

  The frontend implements the following stages:
      `Framer -> Window -> FFT -> FilterBank -> MeanStdDev`

  Also, if stack_left_context > 0, this will further apply:
      `FrameStack -> SubSample(stack_left_context + 1)`

  The FProp input to this layer can either have rank 3 or rank 4 shape:
      [batch_size, timestep, packet_size, channel_count]
      [batch_size, timestep * packet_size, channel_count]

  For compatibility with existing code, 2D [batch_size, timestep] mono shapes
  are also supported.

  In the common case, the packet_size is 1. The 4D variant is accepted for
  glueless interface to input generators that frame their input samples in
  some way. The external framing choice does not influence the operation of
  this instance, but it is accepted.

  TODO(laurenzo): Refactor call sites to uniformly use the 4D variant and
  eliminate fallback logic in this class.

  Only 1 channel is currently supported.
  TODO(laurenzo): Refactor this class to operate on multi-channel inputs.
  """

  @classmethod
  def Params(cls):
    p = super(MelAsrFrontend, cls).Params()
    p.name = 'frontend'
    p.Define('sample_rate', 16000.0, 'Sample rate in Hz')
    p.Define('channel_count', 1, 'Number of channels.')
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
    p.Define('stack_left_context', 0, 'Number of left context frames to stack.')
    return p

  @staticmethod
  def GetConfigFromParams(params):
    """Returns an AsrFrontendConfig namedtuple with vital config settings."""
    subsample_factor = params.num_bins * (params.stack_left_context + 1)
    frame_step = round(params.sample_rate * params.frame_step_ms / 1000.0)
    return AsrFrontendConfig(
        is_null=False,
        src_type='pcm',
        src_pcm_scale=32768.0,
        src_pcm_sample_rate=16000.0,
        output_dim=subsample_factor,
        input_frame_ratio=frame_step * subsample_factor)

  @base_layer.initializer
  def __init__(self, params):
    super(MelAsrFrontend, self).__init__(params)
    p = self.params
    assert p.channel_count == 1, 'Only 1 channel currently supported.'
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
        return tf.signal.hann_window(frame_size, dtype=dtype)

      self._window_fn = _HanningWindow
    else:
      raise ValueError('Illegal value %r for window_fn param' % (p.window_fn,))

  @property
  def window_frame_size(self):
    return self._frame_size

  @property
  def window_frame_step(self):
    return self._frame_step

  def _RemoveChannelDim(self, pcm_audio_data):
    if pcm_audio_data.shape.rank == 3:
      pcm_audio_data = tf.squeeze(pcm_audio_data, 2)
      assert pcm_audio_data.shape.rank == 2, (
          'MelAsrFrontend only supports one channel')
    return pcm_audio_data

  def _ReshapeToMono2D(self, pcm_audio_data, paddings):
    """Reshapes a 3D or 4D input to 2D.

    Since the input to FProp can be 3D or 4D (see class comments), this will
    collapse it back to a 2D, mono shape for internal processing.

    Args:
      pcm_audio_data: 2D, 3D or 4D audio input. See class comments. Must have a
        rank.
      paddings: Original paddings shaped to the first two dims of
        pcm_audio_data.

    Returns:
      Tuple of 2D [batch_size, timestep] mono audio data, new paddings.
    """
    shape = py_utils.GetShape(pcm_audio_data)
    rank = len(shape)
    if rank == 2:
      return pcm_audio_data, paddings
    elif rank == 3:
      # [batch, time, channel]
      with tf.control_dependencies([tf.assert_equal(shape[2], 1)]):
        return tf.squeeze(pcm_audio_data, axis=2), paddings
    elif rank == 4:
      # [batch, time, packet, channel]
      batch_size, orig_time, orig_packet_size, channel = shape
      time = orig_time * orig_packet_size
      with tf.control_dependencies([tf.assert_equal(channel, 1)]):
        pcm_audio_data = tf.reshape(pcm_audio_data, (batch_size, time))
        # Transform paddings into the new time base with a padding per time
        # step vs per packet by duplicating each packet.
        paddings = tf.reshape(
            tf.tile(tf.expand_dims(paddings, axis=2), [1, 1, orig_packet_size]),
            (batch_size, time))
        return pcm_audio_data, paddings
    else:
      raise ValueError('Illegal pcm_audio_data shape')

  def FProp(self, theta, input_batch):
    """Perform signal processing on a sequence of PCM data.

    NOTE: This implementation does not currently support paddings, and they
    are accepted for compatibility with the super-class.

    TODO(laurenzo): Rework this to support paddings.

    Args:
      theta: Layer theta.
      input_batch: PCM input map:

        - 'src_inputs': int16 or float32 tensor of PCM audio data, scaled to
          +/-32768 (versus [-1..1)!). See class comments for supported input
          shapes.
        - 'paddings': per frame 0/1 paddings. Shaped: [batch, frame].
    Returns:
      NestedMap of encoder inputs which can be passed directly to a
      compatible encoder and contains:

        - 'src_inputs': inputs to the encoder, minimally of shape
          [batch, time, ...].
        - 'paddings': a 0/1 tensor of shape [batch, time].
    """

    pcm_audio_data, pcm_audio_paddings = self._ReshapeToMono2D(
        input_batch.src_inputs, input_batch.paddings)

    mel_spectrogram, mel_spectrogram_paddings = self._FPropChunk(
        theta, pcm_audio_data, pcm_audio_paddings)

    mel_spectrogram, mel_spectrogram_paddings = self._PadAndReshapeSpec(
        mel_spectrogram, mel_spectrogram_paddings)

    return py_utils.NestedMap(
        src_inputs=mel_spectrogram, paddings=mel_spectrogram_paddings)

  def _PadAndReshapeSpec(self, mel_spectrogram, mel_spectrogram_paddings):
    p = self.params
    batch_size = py_utils.GetShape(mel_spectrogram)[0]
    # Stack and sub-sample. Only subsampling with a stride of the stack size
    # is supported.
    if p.stack_left_context > 0:
      # Since left context is leading, pad the left by duplicating the first
      # frame.
      stack_size = 1 + p.stack_left_context
      mel_spectrogram = tf.concat(
          [mel_spectrogram[:, 0:1, :]] * p.stack_left_context +
          [mel_spectrogram],
          axis=1)
      mel_spectrogram_paddings = tf.concat(
          [mel_spectrogram_paddings[:, 0:1]] * p.stack_left_context +
          [mel_spectrogram_paddings],
          axis=1)

      # Note that this is the maximum number of frames. Actual frame count
      # depends on padding.
      stacked_frame_dim = tf.shape(mel_spectrogram)[1] // stack_size
      mel_spectrogram = tf.reshape(
          mel_spectrogram[:, 0:(stack_size) * stacked_frame_dim, :],
          [batch_size, stacked_frame_dim, stack_size * p.num_bins])
      # After stacking paddings, pad if any source frame was padded.
      # Stacks into [batch_size, stacked_frame_dim, stack_size] like the
      # spectrogram stacking above, and then reduces the stack_size dim
      # to the max (effectively, making padding = 1.0 if any of the pre-stacked
      # frames were 1.0). Final shape is [batch_size, stacked_frame_dim].
      mel_spectrogram_paddings = tf.reshape(
          mel_spectrogram_paddings[:, 0:(stack_size) * stacked_frame_dim],
          [batch_size, stacked_frame_dim, stack_size])
      mel_spectrogram_paddings = tf.reduce_max(mel_spectrogram_paddings, axis=2)

    # Add feature dim. Shape = [batch, time, features, 1]
    mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
    return mel_spectrogram, mel_spectrogram_paddings

  def _ApplyPreemphasis(self, framed_signal):
    p = self.params
    preemphasized = (
        framed_signal[:, :, 1:] - p.preemph * framed_signal[:, :, 0:-1])
    return preemphasized

  def _GetMelPadding(self, pcm_audio_paddings):
    p = self.params
    # shape: [batch, time, _frame_size]
    framed_paddings = tf.signal.frame(pcm_audio_paddings, self._frame_size,
                                      self._frame_step, p.pad_end)
    # Pad spectrograms that have any padded frames.
    mel_spectrogram_paddings = tf.reduce_max(framed_paddings, axis=2)
    return mel_spectrogram_paddings

  def _FPropChunk(self, theta, pcm_audio_chunk, pcm_audio_paddings):
    p = self.params
    pcm_audio_chunk = tf.cast(pcm_audio_chunk, tf.float32)
    # shape: [batch, time, _frame_size]
    framed_signal = tf.signal.frame(pcm_audio_chunk, self._frame_size,
                                    self._frame_step, p.pad_end)

    # Pre-emphasis.
    if p.preemph != 1.0:
      preemphasized = self._ApplyPreemphasis(framed_signal)
    else:
      preemphasized = framed_signal[:-1]

    # Noise.
    if p.noise_scale > 0.0:
      noise_signal = tf.random_normal(
          tf.shape(preemphasized),
          stddev=p.noise_scale,
          mean=0.0,
          seed=p.random_seed)
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
    return mel_spectrogram_norm, self._GetMelPadding(pcm_audio_paddings)

  def _MelSpectrogram(self, signal):
    """Computes the mel spectrogram from a waveform signal.

    Args:
      signal: f32 Tensor, shaped [batch_size, num_samples]

    Returns:
      features: f32 Tensor, shaped [batch_size, num_frames, mel_channels]
    """
    p = self.params
    # FFT.
    real_frequency_spectrogram = tf.signal.rfft(signal, [self._fft_size])
    magnitude_spectrogram = tf.abs(real_frequency_spectrogram)

    # Shape of magnitude_spectrogram is num_frames x (fft_size/2+1)
    # Mel_weight is [num_spectrogram_bins, num_mel_bins]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
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
