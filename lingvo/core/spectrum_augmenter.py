# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Lingvo layers that used for spectrum augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo.core import base_layer
from lingvo.core import py_utils
from six.moves import range
import tensorflow as tf


class SpectrumAugmenter(base_layer.BaseLayer):
  """Performs data augmentation as according to the SpecAug paper.

  https://arxiv.org/pdf/1904.08779.pdf
  """

  @classmethod
  def Params(cls):
    p = super(SpectrumAugmenter, cls).Params()
    p.Define('freq_mask_max_bins', 15,
             'Maximum number of frequency bins of frequency masking.')
    p.Define('freq_mask_count', 1,
             'Number of times we apply masking on the frequency axis.')
    p.Define('time_mask_max_frames', 50,
             'Maximum number of frames of time masking.')
    p.Define('time_mask_count', 1,
             'Number of times we apply masking on the time axis.')
    p.Define('time_mask_max_ratio', 1.0,
             'Maximum portion allowed for time masking.')
    p.Define('use_noise', False, 'Whether to noisify the time masked region.')
    p.Define('unstack', False,
             'Whether to unstack features before applying SpecAugment')
    p.Define('stack_height', 3, 'Number of frames stacked on top of each other')
    if p.unstack:
      assert (p.stack_height == p.left_content + 1 + p.right_context ==
              p.frame_stride)

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SpectrumAugmenter, self).__init__(params)
    p = self.params
    # TODO(ngyuzh): adding time warping.
    assert p.freq_mask_max_bins > -1
    assert p.time_mask_max_frames > -1

  def _GetMask(self,
               batch_size,
               max_length,
               choose_range,
               mask_size,
               dtype=tf.float32,
               max_ratio=1.0):
    """Returns a fixed size mask starting from a random position.

    In this function:
      1) Sample random lengths less than max_length with shape (batch_size,).
      2) Truncate lengths to a max of range * max_ratio, so that each mask is
         fully contained within the corresponding sequence.
      3) Random sample start points with in (choose_range - lengths).
      4) Return a mask where (lengths - start points) * mask_size are all zeros.

    Args:
      batch_size: batch size.
      max_length: Maximum number of allowed consecutive masked entries.
      choose_range: Range within which the masked entries must lie.
      mask_size: Size of the mask.
      dtype: Data type.
      max_ratio: Maximum portion of the entire range allowed to be masked.

    Returns:
      mask: a fixed size mask starting from a random position with shape
      [batch_size, seq_len].
    """
    p = self.params
    # TODO(ngyuzh): should length distribution depend on utterance length?
    # Choose random masked length
    max_length = tf.random.uniform((batch_size,),
                                   maxval=max_length,
                                   dtype=tf.int32,
                                   seed=p.random_seed)
    # Make sure the sampled length was smaller than max_ratio * length_bound.
    # Note that sampling in this way was biased
    # (shorter sequence may over-masked.)
    length_bound = tf.cast(choose_range, dtype=dtype)
    length_bound = tf.cast(max_ratio * length_bound, dtype=tf.int32)
    length = tf.minimum(max_length, tf.maximum(length_bound, 1))
    # Choose starting point
    random_start = tf.random.uniform((batch_size,),
                                     maxval=1.0,
                                     seed=p.random_seed)
    start_with_in_valid_range = random_start * tf.cast(
        (choose_range - length + 1), dtype=dtype)
    start = tf.cast(start_with_in_valid_range, tf.int32)
    end = start + length - 1

    # Shift starting and end point by small value
    delta = tf.constant(0.1)
    start = tf.expand_dims(tf.cast(start, dtype) - delta, -1)
    end = tf.expand_dims(tf.cast(end, dtype) + delta, -1)

    # Construct mask
    diagonal = tf.tile(
        tf.expand_dims(tf.cast(tf.range(mask_size), dtype=dtype), 0),
        [batch_size, 1])
    mask = 1.0 - tf.cast(
        tf.logical_and(diagonal < end, diagonal > start), dtype=dtype)
    if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
      mask = tf.cast(mask, p.fprop_dtype)
    return mask

  def _FrequencyMask(self, inputs, num_freq=80, dtype=tf.float32):
    """Applies frequency masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      num_freq: Number of frequencies.
      dtype: Data type.

    Returns:
      Inputs with random frequency masking applied.
    """

    # If maximum mask length is zero, do nothing
    p = self.params
    if p.freq_mask_max_bins == 0:
      return inputs

    # Create masks in frequency direction and apply
    block_arrays = self._GetMask(
        tf.shape(inputs)[0],
        p.freq_mask_max_bins,
        choose_range=num_freq,
        mask_size=num_freq,
        dtype=dtype)
    outputs = tf.einsum('bxyc,by->bxyc', inputs, block_arrays)

    return outputs

  def _TimeMask(self,
                inputs,
                seq_lengths,
                max_ratio=1.0,
                time_length=2560,
                noisify=False,
                dtype=tf.float32):
    """Applies time masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      seq_lengths: The actual sequence lengths which mask been sampled of shape
        (batch_size,).
      max_ratio: Maximum portion of the utterance allowed to be time-masked.
      time_length: Total length of time series.
      noisify: whether to noisify the masked out regions.
      dtype: Data type.

    Returns:
      Inputs with random time masking applied.
    """
    p = self.params
    # If maximum mask length is zero, do nothing
    if p.time_mask_max_frames == 0:
      return inputs
    seq_lengths = tf.cast(seq_lengths, tf.int32)

    # Create masks in time direction and apply
    block_arrays = self._GetMask(
        tf.shape(inputs)[0],
        p.time_mask_max_frames,
        choose_range=seq_lengths,
        mask_size=time_length,
        dtype=dtype,
        max_ratio=max_ratio)

    outputs = tf.einsum(
        'bxyc,bx->bxyc', inputs, block_arrays, name='einsum_formasking')
    if noisify:
      # Sample noise with standard deviation with factor * 0.1 + 0.0001
      # TODO(ngyuzh): Make sure this won't affect EOS.
      factor = tf.random_uniform((),
                                 minval=1.0,
                                 maxval=2.0,
                                 dtype=dtype,
                                 seed=p.random_seed)
      stddev = factor * 0.1 + 0.0001
      noise = tf.random.normal(
          [tf.shape(inputs)[0],
           tf.shape(inputs)[1],
           tf.shape(inputs)[2]],
          stddev=stddev,
          seed=p.random_seed)
      if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
        noise = tf.cast(noise, p.fprop_dtype)
      outputs_mask = tf.einsum(
          'bxy,bx->bxy',
          noise,
          1.0 - block_arrays,
          name='einsum_fornoisymasking')
      outputs = outputs + tf.expand_dims(outputs_mask, -1)
    return outputs

  def UnstackFeatures(self, src_inputs, src_paddings):
    """Unstacks src_input and src_paddings based off stack height."""
    sh = self.params.stack_height
    # TODO(ngyuzh) Change to py_utils.GetShape
    bs, old_series_length, _, channels = py_utils.GetShape(src_inputs)
    unstacked_series_length = old_series_length * sh
    src_inputs = tf.reshape(src_inputs,
                            [bs, unstacked_series_length, -1, channels])
    content = 1 - src_paddings
    lengths = tf.cast(sh * tf.reduce_sum(content, axis=1), tf.int32)
    mask = tf.sequence_mask(lengths, maxlen=unstacked_series_length)
    src_paddings = 1 - tf.cast(mask, tf.int32)
    return src_inputs, src_paddings

  def _AugmentationNetwork(self, series_length, num_freq, inputs, paddings):
    """Returns augmented features.

    Args:
      series_length: Total length of time series.
      num_freq: Number of frequencies.
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      paddings: Batch of padding vectors of shape (batch_size, time_length).

    Returns:
      output: Batch of output features of shape
              (batch_size, time_length, num_freq, channels)
              obtained by applying random augmentations to inputs.
    """
    p = self.params
    dtype = p.dtype

    # Unstack the features.
    if p.unstack:
      stacked_series_length = series_length
      inputs, paddings = self.UnstackFeatures(inputs, paddings)
      num_freq //= p.stack_height
      series_length *= p.stack_height

    lengths = tf.reduce_sum(1 - paddings, 1)
    for _ in range(p.time_mask_count):
      inputs = self._TimeMask(
          inputs,
          lengths,
          max_ratio=p.time_mask_max_ratio,
          time_length=series_length,
          noisify=p.use_noise,
          dtype=dtype)
    for _ in range(p.freq_mask_count):
      inputs = self._FrequencyMask(inputs, num_freq=num_freq, dtype=dtype)

    # Restack the features after applying specaugment.
    if p.unstack:
      inputs = tf.reshape(
          inputs,
          [tf.shape(inputs)[0], stacked_series_length, -1,
           tf.shape(inputs)[3]])

    return inputs

  def FProp(self, theta, inputs, paddings):
    """Applies data augmentation by randomly mask spectrum in inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: A tensor of shape [batch, time, freq, num_channels].
      paddings: A 0/1 tensor of shape [batch, time].

    Returns:
      augmented_inputs: An tensor of shape [batch, time, freq, num_channels].
      paddings: A 0/1 tensor of shape [batch, time].
    """
    _, series_length, num_freq, _ = py_utils.GetShape(inputs)
    augmented_inputs = self._AugmentationNetwork(series_length, num_freq,
                                                 inputs, paddings)
    return augmented_inputs, paddings
