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

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils

_SPECAUGMENT_ARGS = (
    'freq_mask_max_bins',
    'freq_mask_count',
    'time_mask_max_frames',
    'time_mask_count',
    'time_mask_max_ratio',
    'time_masks_per_frame',
    'use_dynamic_time_mask_max_frames',
)


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
    p.Define(
        'time_mask_max_frames', 50, 'Maximum number of frames of time masking. '
        'Overridden when use_dynamic_time_mask_max_frames = True.')
    p.Define(
        'time_mask_count', 1,
        'Number of times we apply masking on the time axis. '
        'Acts as upper-bound when time_masks_per_frame > 0.')
    p.Define('time_mask_max_ratio', 1.0,
             'Maximum portion allowed for time masking.')
    p.Define(
        'time_masks_per_frame', 0.0,
        'Ratio of number of time masks to be applied against '
        'the number of frames. If > 0, multiplicity of the time mask '
        'is determined by time_masks_per_frame * utterance_length.')
    p.Define(
        'use_dynamic_time_mask_max_frames', False,
        'If true, time_mask_max_frames is determined by '
        'time_mask_max_ratio * utterance_length.')
    p.Define('use_noise', False, 'Whether to noisify the time masked region.')
    p.Define('gaussian_noise', False, 'Use Gaussian distribution for noise.')
    p.Define('unstack', False,
             'Whether to unstack features before applying SpecAugment.')
    p.Define('stack_height', 3,
             'Number of frames stacked on top of each other.')
    p.Define(
        'domain_ids', [0],
        'If domain ids was given, this parameters describe which domain '
        'will be augmented, e.g. '
        'p.domain_ids = [2, 7, 1] '
        'p.time_mask_count = [1, 2, 0] '
        'implies domain 2 will have 1, 7 has 2 and 1 has 0 time masks. '
        'All other domain will not augmented if it exists.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SpectrumAugmenter, self).__init__(params)
    p = self.params
    num_domains = len(p.domain_ids)
    for field in _SPECAUGMENT_ARGS:
      v = getattr(p, field)
      if isinstance(v, (list, tuple)):
        assert len(v) == num_domains
      else:
        setattr(p, field, [v] * num_domains)
    # TODO(ngyuzh): adding time warping.
    assert p.freq_mask_max_bins[0] > -1
    assert p.time_mask_max_frames[0] > -1

  def _GetMask(self,
               batch_size,
               choose_range,
               mask_size,
               max_length=None,
               masks_per_frame=0.0,
               multiplicity=1,
               dtype=tf.float32,
               max_ratio=1.0):
    """Returns fixed size multi-masks starting from random positions.

    A multi-mask is a mask obtained by applying multiple masks.

    This function when max_length is given:
      1) Sample random mask lengths less than max_length with shape
         (batch_size, multiplicity).
      2) Truncate lengths to a max of (choose_range * max_ratio),
         so that each mask is fully contained within the corresponding sequence.
      3) Random sample start points of shape (batch_size, multiplicity)
         with in (choose_range - lengths).
      4) For each batch, multiple masks (whose number is given by the
         multiplicity) are constructed.
      5) Return a mask of shape (batch_size, mask_size) where masks are
         obtained by composing the masks constructed in step 4).
         If masks_per_frame > 0, the number is given by
         min(masks_per_frame * choose_range, multiplicity).
         If not, all the masks are composed. The masked regions are set to zero.

    This function when max_length is not given:
      1) Sample random mask lengths less than (choose_range * max_ratio)
         with shape (batch_size, multiplicity).
      2) Proceed to steps 3), 4) and 5) of the above.

    Args:
      batch_size: Batch size. Integer number.
      choose_range: Range within which the masked entries must lie. Tensor of
        shape (batch_size,).
      mask_size: Size of the mask. Integer number.
      max_length: Maximum number of allowed consecutive masked entries. Integer
        number or None.
      masks_per_frame: Number of masks per frame. Float number. If > 0, the
        multiplicity of the mask is set to be masks_per_frame * choose_range.
      multiplicity: Maximum number of total masks. Integer number.
      dtype: Data type.
      max_ratio: Maximum portion of the entire range allowed to be masked. Float
        number.

    Returns:
      mask: a fixed size multi-mask starting from a random position with shape
      (batch_size, mask_size).
    """
    p = self.params
    # Non-empty random seed values are only used for testing
    # seed_1 and seed_2 are set separately to avoid correlation of
    # mask size and mask position.
    if p.random_seed:
      seed_1 = p.random_seed + 1
      seed_2 = 2 * p.random_seed
    else:
      seed_1 = p.random_seed
      seed_2 = p.random_seed
    # Sample lengths for multiple masks.
    if max_length and max_length > 0:
      max_length = tf.broadcast_to(tf.cast(max_length, dtype), (batch_size,))
    else:
      max_length = tf.cast(choose_range, dtype=dtype) * max_ratio
    masked_portion = tf.random.uniform((batch_size, multiplicity),
                                       minval=0.0,
                                       maxval=1.0,
                                       dtype=dtype,
                                       seed=seed_1)
    masked_frame_size = tf.einsum('b,bm->bm', max_length, masked_portion)
    masked_frame_size = tf.cast(masked_frame_size, dtype=tf.int32)
    # Make sure the sampled length was smaller than max_ratio * length_bound.
    # Note that sampling in this way was biased
    # (shorter sequence may over-masked.)
    choose_range = tf.expand_dims(choose_range, -1)
    choose_range = tf.tile(choose_range, [1, multiplicity])
    length_bound = tf.cast(choose_range, dtype=dtype)
    length_bound = tf.cast(max_ratio * length_bound, dtype=tf.int32)
    length = tf.minimum(masked_frame_size, tf.maximum(length_bound, 1))

    # Choose starting point.
    random_start = tf.random.uniform((batch_size, multiplicity),
                                     maxval=1.0,
                                     seed=seed_2)
    start_with_in_valid_range = random_start * tf.cast(
        (choose_range - length + 1), dtype=dtype)
    start = tf.cast(start_with_in_valid_range, tf.int32)
    end = start + length - 1

    # Shift starting and end point by small value.
    delta = tf.constant(0.1)
    start = tf.expand_dims(tf.cast(start, dtype) - delta, -1)
    start = tf.tile(start, [1, 1, mask_size])
    end = tf.expand_dims(tf.cast(end, dtype) + delta, -1)
    end = tf.tile(end, [1, 1, mask_size])

    # Construct pre-mask of shape (batch_size, multiplicity, mask_size).
    diagonal = tf.expand_dims(
        tf.expand_dims(tf.cast(tf.range(mask_size), dtype=dtype), 0), 0)
    diagonal = tf.tile(diagonal, [batch_size, multiplicity, 1])
    pre_mask = tf.cast(
        tf.logical_and(diagonal < end, diagonal > start), dtype=dtype)

    # Sum masks with appropriate multiplicity.
    if masks_per_frame > 0:
      multiplicity_weights = tf.tile(
          tf.expand_dims(tf.range(multiplicity, dtype=dtype), 0),
          [batch_size, 1])
      multiplicity_tensor = masks_per_frame * tf.cast(choose_range, dtype=dtype)
      multiplicity_weights = tf.cast(
          multiplicity_weights < multiplicity_tensor, dtype=dtype)
      pre_mask = tf.einsum('bmt,bm->bt', pre_mask, multiplicity_weights)
    else:
      pre_mask = tf.reduce_sum(pre_mask, 1)
    mask = tf.cast(1.0 - tf.cast(pre_mask > 0, dtype=dtype), dtype=dtype)

    if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
      mask = tf.cast(mask, p.fprop_dtype)

    return mask

  def _FrequencyMask(self,
                     inputs,
                     dtype=tf.float32,
                     domain_id_index=0):
    """Applies frequency masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      dtype: Data type.
      domain_id_index: domain id index.

    Returns:
      Inputs with random frequency masking applied.
    """
    p = self.params

    # Mask parameters.
    freq_mask_max_bins = p.freq_mask_max_bins[domain_id_index]
    multiplicity = p.freq_mask_count[domain_id_index]

    # If masking length or count is zero, do nothing.
    if freq_mask_max_bins == 0 or multiplicity == 0:
      return inputs

    # Arguments to pass to mask generator.
    batch_size, _, num_freq, _ = py_utils.GetShape(inputs)
    choose_range = tf.cast(
        tf.broadcast_to(num_freq, (batch_size,)), dtype=tf.int32)
    # Create masks in frequency direction and apply.
    block_arrays = self._GetMask(
        tf.shape(inputs)[0],
        choose_range=choose_range,
        mask_size=num_freq,
        max_length=freq_mask_max_bins,
        masks_per_frame=0.0,
        multiplicity=multiplicity,
        dtype=dtype,
        max_ratio=1.0)
    outputs = tf.einsum('bxyc,by->bxyc', inputs, block_arrays)

    return outputs

  def _TimeMask(self,
                inputs,
                seq_lengths,
                noisify=False,
                gaussian_noise=False,
                dtype=tf.float32,
                domain_id_index=0):
    """Applies time masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      seq_lengths: The actual sequence lengths which mask been sampled of shape
        (batch_size,).
      noisify: Whether to noisify the masked out regions.
      gaussian_noise: Whether to use gaussian noise when noisifying.
      dtype: Data type.
      domain_id_index: domain id index.

    Returns:
      Inputs with random time masking applied.
    """
    p = self.params

    # Get time masking parameters.
    time_mask_max_frames = p.time_mask_max_frames[domain_id_index]
    time_masks_per_frame = p.time_masks_per_frame[domain_id_index]
    use_dynamic_time_mask_max_frames = \
        p.use_dynamic_time_mask_max_frames[domain_id_index]
    multiplicity = p.time_mask_count[domain_id_index]
    max_ratio = p.time_mask_max_ratio[domain_id_index]

    # If maximum mask length is zero, do nothing.
    if ((time_mask_max_frames == 0 and not use_dynamic_time_mask_max_frames) or
        max_ratio <= 0.0):
      return inputs
    if multiplicity == 0:
      return inputs
    seq_lengths = tf.cast(seq_lengths, tf.int32)
    batch_size, time_length, _, _ = py_utils.GetShape(inputs)

    # When using dynamic time mask size, discard upper-bound on
    # maximum allowed frames for time mask.
    if use_dynamic_time_mask_max_frames:
      time_mask_max_frames = None
    # Create masks in time direction and apply.
    block_arrays = self._GetMask(
        batch_size,
        choose_range=seq_lengths,
        mask_size=time_length,
        max_length=time_mask_max_frames,
        masks_per_frame=time_masks_per_frame,
        multiplicity=multiplicity,
        dtype=dtype,
        max_ratio=max_ratio)

    outputs = tf.einsum(
        'bxyc,bx->bxyc', inputs, block_arrays, name='einsum_formasking')
    if noisify:
      # Sample noise with standard deviation with factor * 0.1 + 0.0001
      # TODO(ngyuzh): Make sure this won't affect EOS.
      if gaussian_noise:
        stddev = 1.0
      else:
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
    bs, old_series_length, _, channels = py_utils.GetShape(src_inputs)
    unstacked_series_length = old_series_length * sh
    src_inputs = tf.reshape(src_inputs,
                            [bs, unstacked_series_length, -1, channels])
    content = 1 - src_paddings
    lengths = tf.cast(sh * tf.reduce_sum(content, axis=1), tf.int32)
    mask = tf.sequence_mask(lengths, maxlen=unstacked_series_length)
    src_paddings = 1 - tf.cast(mask, tf.int32)
    return src_inputs, src_paddings

  def _AugmentationNetwork(self,
                           series_length,
                           inputs,
                           paddings,
                           domain_id_index=0):
    """Returns augmented features.

    Args:
      series_length: Total length of time series.
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      paddings: Batch of padding vectors of shape (batch_size, time_length).
      domain_id_index: domain id index.

    Returns:
      Batch of output features of shape (batch_size, time_length, num_freq,
      channels) obtained by applying random augmentations to inputs.
    """
    p = self.params
    dtype = p.dtype

    # Unstack the features.
    if p.unstack:
      inputs, paddings = self.UnstackFeatures(inputs, paddings)

    lengths = tf.reduce_sum(1 - paddings, 1)
    inputs = self._TimeMask(
        inputs,
        lengths,
        noisify=p.use_noise,
        gaussian_noise=p.gaussian_noise,
        dtype=dtype,
        domain_id_index=domain_id_index)
    inputs = self._FrequencyMask(
        inputs, dtype=dtype, domain_id_index=domain_id_index)

    # Restack the features after applying specaugment.
    if p.unstack:
      inputs = tf.reshape(
          inputs, [tf.shape(inputs)[0], series_length, -1,
                   tf.shape(inputs)[3]])

    return inputs

  def FProp(self, theta, inputs, paddings, domain_ids=None):
    """Applies data augmentation by randomly mask spectrum in inputs.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      inputs: A tensor of shape [batch, time, freq, num_channels].
      paddings: A 0/1 tensor of shape [batch, time].
      domain_ids: input domain_ids of shape [batch, time].

    Returns:
      A pair of 2 tensors:

      - augmented_inputs: A tensor of shape [batch, time, freq, num_channels].
      - paddings: A 0/1 tensor of shape [batch, time].
    """
    p = self.params

    batch_size, series_length, _, _ = py_utils.GetShape(inputs)
    if len(p.domain_ids) > 1:
      augmented_inputs = tf.zeros_like(inputs)
      original_inputs = inputs
      for i, domain_id in enumerate(p.domain_ids):
        augmented_domain = self._AugmentationNetwork(
            series_length, inputs, paddings, domain_id_index=i)
        target_domain = tf.cast(
            tf.expand_dims(tf.tile([domain_id], [batch_size]), -1),
            dtype=p.dtype)
        # [batch, time].
        domain_mask = tf.cast(
            tf.equal(domain_ids, target_domain), dtype=p.dtype)
        augmented_domain = tf.einsum(
            'bxyc,bx->bxyc',
            augmented_domain,
            domain_mask,
            name='einsum_domainmasking')
        original_inputs = tf.einsum(
            'bxyc,bx->bxyc',
            original_inputs,
            1.0 - domain_mask,
            name='einsum_domainmasking2')
        augmented_inputs = augmented_domain + augmented_inputs
      augmented_inputs = original_inputs + augmented_inputs
    else:
      augmented_inputs = self._AugmentationNetwork(
          series_length, inputs, paddings, domain_id_index=0)
    return augmented_inputs, paddings
