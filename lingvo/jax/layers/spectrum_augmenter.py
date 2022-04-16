# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Spectrum augmentation layers."""

from typing import Optional, Tuple

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes

NestedMap = py_utils.NestedMap
InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


# TODO(nanxinchen): add time wrap and frequency wrap
class SpectrumAugmenter(base_layer.BaseLayer):
  """Performs data augmentation as according to the SpecAug paper.

    https://arxiv.org/pdf/1904.08779.pdf
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('freq_mask_max_bins', 15,
             'Maximum number of frequency bins of frequency masking.')
    p.Define('freq_mask_count', 1,
             'Number of times we apply masking on the frequency axis.')
    p.Define(
        'use_dynamic_time_mask_max_frames', False,
        'If true, time_mask_max_frames is determined by '
        'time_mask_max_ratio * utterance_length.')
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
        'Ratio of number of time masks to be applied against the number '
        'of frames. If > 0, multiplicity of the time mask is determined by '
        'min(time_masks_per_frame * utterance_length, time_mask_count).')
    return p

  def _get_mask(self,
                batch_size: int,
                choose_range: JTensor,
                mask_size: int,
                global_seed: JTensor,
                max_length: Optional[int] = None,
                masks_per_frame: float = 0.0,
                multiplicity: int = 1,
                max_ratio: float = 1.0) -> JTensor:
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
      global_seed: an integer seed tensor for stateless random ops.
      max_length: Maximum number of allowed consecutive masked entries. Integer
        number or None.
      masks_per_frame: Number of masks per frame. Float number. If > 0, the
        multiplicity of the mask is set to be masks_per_frame * choose_range.
      multiplicity: Maximum number of total masks. Integer number.
      max_ratio: Maximum portion of the entire range allowed to be masked. Float
        number.

    Returns:
      mask: a fixed size multi-mask starting from a random position with shape
      (batch_size, mask_size).
    """
    # Sample lengths for multiple masks.
    if max_length and max_length > 0:
      max_length = jnp.tile(max_length, (batch_size,))
    else:
      max_length = choose_range * max_ratio
    masked_portion = jax.random.uniform(
        key=global_seed,
        shape=(batch_size, multiplicity),
        minval=0.0,
        maxval=1.0)
    masked_frame_size = jnp.einsum('b,bm->bm', max_length,
                                   masked_portion).astype(jnp.int32)
    # Make sure the sampled length was smaller than max_ratio * length_bound.
    # Note that sampling in this way was biased
    # (shorter sequence may over-masked.)
    choose_range = jnp.tile(choose_range[:, None], [1, multiplicity])
    length_bound = (max_ratio * choose_range).astype(jnp.int32)
    length = jnp.minimum(masked_frame_size, jnp.maximum(length_bound, 1))

    # Choose starting point.
    random_start = jax.random.uniform(
        key=global_seed, shape=(batch_size, multiplicity), maxval=1.0)
    start_with_in_valid_range = random_start * (choose_range - length + 1)
    start = start_with_in_valid_range.astype(jnp.int32)
    end = start + length - 1

    # Shift starting and end point by small value.
    delta = 0.1
    start = jnp.expand_dims(start - delta, -1)
    start = jnp.tile(start, [1, 1, mask_size])
    end = jnp.expand_dims(end + delta, -1)
    end = jnp.tile(end, [1, 1, mask_size])

    # Construct pre-mask of shape (batch_size, multiplicity, mask_size).
    diagonal = jnp.expand_dims(jnp.expand_dims(jnp.arange(mask_size), 0), 0)
    diagonal = jnp.tile(diagonal, [batch_size, multiplicity, 1])
    pre_mask = jnp.minimum(diagonal < end, diagonal > start)

    # Sum masks with appropriate multiplicity.
    if masks_per_frame > 0:
      multiplicity_weights = jnp.tile(
          jnp.expand_dims(jnp.arange(multiplicity, dtype=jnp.int32), 0),
          [batch_size, 1])
      multiplicity_tensor = masks_per_frame * choose_range
      multiplicity_weights = (multiplicity_weights <
                              multiplicity_tensor).astype(jnp.int32)
      pre_mask = jnp.einsum('bmt,bm->bt', pre_mask, multiplicity_weights)
    else:
      pre_mask = jnp.einsum('bmt->bt', pre_mask)
    mask = 1.0 - (pre_mask > 0).astype(jnp.int32)

    return mask

  def _time_mask(self, inputs: JTensor, length: JTensor,
                 global_seed) -> JTensor:
    """Applies time masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq).
      length: number of frames in each sample (batch_size,)
      global_seed: an prng key for stateless random ops.

    Returns:
      Inputs with random time masking applied.
    """
    p = self.params

    # Get time masking parameters.
    time_mask_max_frames = p.time_mask_max_frames
    time_masks_per_frame = p.time_masks_per_frame
    use_dynamic_time_mask_max_frames = p.use_dynamic_time_mask_max_frames
    multiplicity = p.time_mask_count
    max_ratio = p.time_mask_max_ratio

    # If maximum mask length is zero, do nothing.
    if ((time_mask_max_frames == 0 and not use_dynamic_time_mask_max_frames) or
        max_ratio <= 0.0):
      return inputs
    if multiplicity == 0:
      return inputs
    batch_size, time_length, _ = inputs.shape

    # When using dynamic time mask size, discard upper-bound on
    # maximum allowed frames for time mask.
    if use_dynamic_time_mask_max_frames:
      time_mask_max_frames = None
    # Create masks in time direction and apply.
    block_arrays = self._get_mask(
        batch_size,
        choose_range=length,
        mask_size=time_length,
        global_seed=global_seed,
        max_length=time_mask_max_frames,
        masks_per_frame=time_masks_per_frame,
        multiplicity=multiplicity,
        max_ratio=max_ratio)

    outputs = jnp.einsum('bxy,bx->bxy', inputs, block_arrays)

    return outputs

  def _frequency_mask(self, inputs: JTensor, global_seed) -> JTensor:
    """Applies frequency masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq).
      global_seed: an prng tensor for stateless random ops.

    Returns:
      Inputs with random frequency masking applied.
    """
    p = self.params

    # Mask parameters.
    freq_mask_max_bins = p.freq_mask_max_bins
    multiplicity = p.freq_mask_count

    # If masking length or count is zero, do nothing.
    if freq_mask_max_bins == 0 or multiplicity == 0:
      return inputs

    # Arguments to pass to mask generator.
    batch_size, _, num_freq = inputs.shape
    choose_range = jnp.tile(num_freq, (batch_size,))
    # Create masks in frequency direction and apply.
    block_arrays = self._get_mask(
        batch_size,
        choose_range=choose_range,
        mask_size=num_freq,
        global_seed=global_seed,
        max_length=freq_mask_max_bins,
        masks_per_frame=0.0,
        multiplicity=multiplicity,
        max_ratio=1.0)

    outputs = jnp.einsum('bxy,by->bxy', inputs, block_arrays)

    return outputs

  def fprop(self, inputs: JTensor,
            paddings: JTensor) -> Tuple[JTensor, JTensor]:
    """Applies data augmentation by randomly masking values in the spectrum.

    Args:
      inputs: A tensor of shape [batch, length, channels].
      paddings: A 0/1 tensor of shape [batch, length].

    Returns:
      A pair <new_inputs, mask>:
      new_inputs: A tensor of shape [batch, length, channels].
      paddings: A 0/1 tensor of shape [batch, length].
    """
    lengths = jnp.einsum('bh->b', 1 - paddings).astype(jnp.int32)

    prng_key = base_layer.next_prng_key()
    inputs = self._time_mask(inputs, lengths, global_seed=prng_key)
    prng_key = base_layer.next_prng_key()
    inputs = self._frequency_mask(inputs, global_seed=prng_key)

    return inputs, paddings
