# Lint as: python3
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

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils

_SPECAUGMENT_ARGS = (
    'freq_mask_max_bins',
    'freq_mask_count',
    'use_dynamic_time_mask_max_frames',
    'time_mask_max_frames',
    'time_mask_count',
    'time_mask_max_ratio',
    'time_masks_per_frame',
    'freq_warp_max_bins',
    'time_warp_bound',
    'time_warp_max_frames',
    'time_warp_max_ratio',
    'freq_noise_max_stddev',
)


def _random_uniform_op(use_stateless_op):
  return tf.random.stateless_uniform if use_stateless_op else tf.random.uniform


def _random_normal_op(use_stateless_op):
  return tf.random.stateless_normal if use_stateless_op else tf.random.normal


def _global_seed_from_inputs(input_floats):
  """Generates a random seed tensor based on input floats and mode key.

  Args:
    input_floats: a set of float input tensors that are derived from the input
      data (for example, input tokens). The important thing is that these are
      usually different for each batch.

  Returns:
    A tensor of shape=[2] with integer seed tensors derived from the inputs.
  """
  timestamp = tf.math.floormod(
      tf.cast(1000.0 * tf.timestamp(), dtype=tf.int64), 10000000000)
  input_sum = tf.cast(tf.reduce_sum(tf.math.abs(input_floats)), dtype=tf.int64)
  return tf.stack([timestamp + input_sum, timestamp - input_sum], axis=-1)


def _hat(x):
  """Hat function.

  The hat function is a piecewise linear function defined such that
    1) x < -1: _hat(x) = 0
    2) -1 <= x < 0: _hat(x) = x + 1
    3) 0 <= x < 1: _hat(x) = -x + 1
    4) x > 1 : _hat(x) = 0

  Args:
    x: A tensor.

  Returns:
    Tensor obtained by element-wise application of the hat function.
  """
  return tf.nn.relu(x + 1) - 2 * tf.nn.relu(x) + tf.nn.relu(x - 1)


class SpectrumAugmenter(base_layer.BaseLayer):
  """Performs data augmentation as according to the SpecAug paper.

  https://arxiv.org/pdf/1904.08779.pdf
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('freq_mask_max_bins', 15,
             'Maximum number of frequency bins of frequency masking.')
    p.Define('freq_mask_count', 1,
             'Number of times we apply masking on the frequency axis.')
    # TODO(danielspark): Deprecate 'use_dynamic_time_mask_max_frames' and
    # introduce enum parameter to replace it.
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
    p.Define('freq_warp_max_bins', 0,
             'Maximum number of frequency bins for shifting in freq warping.')
    p.Define(
        'time_warp_bound', 'static',
        'To be set to either `dynamic` or `static`. '
        'If `dynamic`, time warp bound is determined by '
        'time_warp_max_ratio * utterance_length. '
        'If `static`, time warp bound is determined by '
        'min(time_warp_max_frames, time_warp_max_ratio * utterance_length).')
    p.Define('time_warp_max_frames', 0,
             'Maximum number of frames for shifting in time warping.')
    p.Define('time_warp_max_ratio', 0.0,
             'Maximum portion of frames for shifting in time warping.')
    p.Define('use_noise', False, 'Whether to noisify the time masked region.')
    p.Define('gaussian_noise', False, 'Use Gaussian distribution for noise.')
    p.Define(
        'freq_noise_max_stddev', 0.0,
        'Maximum stddev of frequency noise. stddev is sampled by uniform '
        'probability in range(0, freq_noise_max_stddev)')
    p.Define(
        'freq_noise_warmup_steps', 0, 'Number of warm-up steps, in which the '
        'freq_noise stddev increases linearly.')
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
    p.Define(
        'use_input_dependent_random_seed', False,
        'Whether to use stateless random TensorFlow ops, with seeds'
        'determined by the input features. This feature is necessary for'
        'applications including federated learning.')
    return p

  def __init__(self, params):
    super().__init__(params)
    self.SetVariableFree()
    p = self.params
    num_domains = len(p.domain_ids)
    for field in _SPECAUGMENT_ARGS:
      v = getattr(p, field)
      if isinstance(v, (list, tuple)):
        assert len(v) == num_domains, (
            'Length: %d of field: %s does not match total domains: %d' %
            (len(v), field, num_domains))
      else:
        setattr(p, field, [v] * num_domains)
    assert p.freq_mask_max_bins[0] > -1
    assert p.time_mask_max_frames[0] > -1
    assert p.freq_warp_max_bins[0] > -1
    assert p.time_warp_max_frames[0] > -1
    assert p.freq_noise_max_stddev[0] >= 0.0

  def EinsumBBmBm(self, a, b, name=None):
    return tf.einsum('b,bm->bm', a, b, name=name)

  def EinsumBmtBmBt(self, a, b, name=None):
    return tf.einsum('bmt,bm->bt', a, b, name=name)

  def EinsumBxycByBxyc(self, a, b, name=None):
    return tf.einsum('bxyc,by->bxyc', a, b, name=name)

  def EinsumBxycBxBxyc(self, a, b, name=None):
    return tf.einsum('bxyc,bx->bxyc', a, b, name=name)

  def EinsumBxyBxBxy(self, a, b, name=None):
    return tf.einsum('bxy,bx->bxy', a, b, name=name)

  def EinsumBxycBzxBzyc(self, a, b, name=None):
    return tf.einsum('bxyc,bzx->bzyc', a, b, name=name)

  def EinsumBxycBzyBxzc(self, a, b, name=None):
    return tf.einsum('bxyc,bzy->bxzc', a, b, name=name)

  @property
  def augment_weight(self):
    # TODO(b/180135215): apply it to other augmentations.
    p = self.params
    if p.freq_noise_warmup_steps == 0:
      return tf.cast(tf.constant(1.), p.dtype)
    global_step = tf.cast(py_utils.GetGlobalStep(), p.dtype)
    augment_warmup_steps = tf.cast(p.freq_noise_warmup_steps, p.dtype)
    return tf.minimum(global_step, augment_warmup_steps) / augment_warmup_steps

  def _GetMask(self,
               batch_size,
               choose_range,
               mask_size,
               global_seed,
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
      global_seed: an integer seed tensor for stateless random ops.
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
    # Non-empty random seed values are only used for testing or when using
    # stateless random ops. seed_1 and seed_2 are set separately to avoid
    # correlation of mask size and mask position.
    if p.use_input_dependent_random_seed:
      seed_1 = global_seed + 1
      seed_2 = global_seed + 2
    elif p.random_seed:
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
    random_uniform = _random_uniform_op(p.use_input_dependent_random_seed)
    masked_portion = random_uniform(
        shape=(batch_size, multiplicity),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
        seed=seed_1)
    masked_frame_size = self.EinsumBBmBm(max_length, masked_portion)
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
    random_start = random_uniform(
        shape=(batch_size, multiplicity), maxval=1.0, seed=seed_2)
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
        tf.math.logical_and(diagonal < end, diagonal > start), dtype=dtype)

    # Sum masks with appropriate multiplicity.
    if masks_per_frame > 0:
      multiplicity_weights = tf.tile(
          tf.expand_dims(tf.range(multiplicity, dtype=dtype), 0),
          [batch_size, 1])
      multiplicity_tensor = masks_per_frame * tf.cast(choose_range, dtype=dtype)
      multiplicity_weights = tf.cast(
          multiplicity_weights < multiplicity_tensor, dtype=dtype)
      pre_mask = self.EinsumBmtBmBt(pre_mask, multiplicity_weights)
    else:
      pre_mask = tf.reduce_sum(pre_mask, 1)
    mask = tf.cast(1.0 - tf.cast(pre_mask > 0, dtype=dtype), dtype=dtype)

    if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
      mask = tf.cast(mask, p.fprop_dtype)

    return mask

  def _GetWarpMatrix(self,
                     batch_size,
                     choose_range,
                     matrix_size,
                     global_seed,
                     max_warp_frames=None,
                     dtype=tf.float32,
                     max_ratio=1.0):
    """Returns warp matrices starting from random positions.

    In this function when max_warp_frames != None:
      1) Sample random warp displacements from the interval
         [-max_warp_frames, max_warp_frames) to yield shift tensor
         with shape (batch_size,).
      2) Truncate lengths to a maximum magnitude of (choose_range * max_ratio),
         so that each shift is fully contained within the
         corresponding sequence.
      3) Random sample origin points of shape (batch_size, multiplicity)
         with in [shift, choose_range - shift).
      4) Return a batch of 1-D linear maps that fix the boundary points and
         shift the origin point by the shift.

    When max_warp_frames == None:
      1) Sample random warp displacements with magnitudes less than
         (choose_range * max_ratio) to yield shift tensor with
         shape (batch_size,).
      2) Proceed through steps 3), 4).

    Args:
      batch_size: Batch size. Integer number.
      choose_range: Range within which the warp reference points must lie.
        Tensor of shape (batch_size,).
      matrix_size: Dimension of vector space warp matrix is applied to. Integer
        number.
      global_seed: an integer seed tensor for stateless random ops.
      max_warp_frames: Upper-bound on the warp distance. Integer or None.
      dtype: Data type.
      max_ratio: Maximum ratio between the shift distance and choose_range.
        Float number.

    Returns:
      warp_matrix: An array of fixed size warp matrices with shape
      (batch_size, matrix_size, matrix_size).
    """
    p = self.params
    # Non-empty random seed values are only used for testing or when using
    # stateless random ops. seed_3, seed_4, and seed_5 are set separately to
    # avoid correlation of warp magnitude and origin position.
    if p.use_input_dependent_random_seed:
      seed_3 = global_seed + 3
      seed_4 = global_seed + 4
      seed_5 = global_seed + 5
    elif p.random_seed:
      seed_3 = p.random_seed - 1
      seed_4 = p.random_seed - 1
      seed_5 = 2 * p.random_seed + 1
    else:
      seed_3 = p.random_seed
      seed_4 = p.random_seed
      seed_5 = p.random_seed

    choose_range_dtype = tf.cast(choose_range, dtype=dtype)
    length_upper_bound = tf.cast(max_ratio * choose_range_dtype, dtype=tf.int32)
    # Set shift length.

    random_uniform = _random_uniform_op(p.use_input_dependent_random_seed)

    if max_warp_frames and max_warp_frames > 0:
      shift = random_uniform(
          shape=(batch_size,),
          minval=-1 * max_warp_frames,
          maxval=max_warp_frames + 1,
          dtype=tf.int32,
          seed=seed_3)
    else:
      random_ratio = random_uniform(
          shape=(batch_size,),
          minval=-1.0,
          maxval=1.0,
          dtype=dtype,
          seed=seed_4)
      shift = tf.cast(random_ratio * tf.cast(length_upper_bound, dtype=dtype),
                      tf.int32)
    # Make sure the sampled length was smaller than max_ratio * length_bound.
    # Note that sampling in this way is biased.
    # (Shorter sequence may over-masked.)
    final_shift = tf.maximum(-length_upper_bound,
                             tf.minimum(shift, length_upper_bound))
    # Choose origin anchor point.
    mid_range = tf.cast(choose_range, dtype=tf.int32)
    mid_range = tf.maximum(choose_range - 2, 0)
    random_origin = random_uniform(shape=(batch_size,), maxval=1.0, seed=seed_5)
    origin_with_in_valid_range = random_origin * tf.cast(mid_range, dtype=dtype)
    origin = tf.cast(origin_with_in_valid_range, tf.int32) + 1
    # Set destination point of the origin anchor point under the warp map.
    destination = origin + final_shift
    # Cast origin and destination.
    origin = tf.cast(origin, dtype=dtype)
    destination = tf.cast(destination, dtype=dtype)

    return self._ConstructWarpMatrix(
        batch_size=batch_size,
        matrix_size=matrix_size,
        origin=origin,
        destination=destination,
        choose_range=choose_range_dtype,
        dtype=dtype)

  def _ConstructWarpMatrix(self, batch_size, matrix_size, origin, destination,
                           choose_range, dtype):
    """Returns warp matrices according to origin, destination and choose_range.

    This function constructs a batch of warp matrices which maps the batch
    of origin points to the batch of destination points with fixed boundary
    coordinates at 0 and choose_range.

    The warping function, defined by the origin anchor point `origin`,
    the destination of the origin anchor point `destination` and the
    length of the domain in the warping axis `choose_range` is a piecewise
    linear map that fixes the points 0 and `choose_range` and maps
    `origin` to `destination`.

    For the warping matrix to be non-singular, destination must lie in the
    range 1<= destination <= choose_range - 1, so a destination
    out of this range is adjusted to be in this range before the warping
    matrix is constructed.

    The warping map can be explicitly written by first defining the slopes:
      1) slope_0 = origin / destination.
      2) slope_1 = (choose_range - origin) / (choose_range - destination).
      3) slope_2 = 1.0.

    Then the origin point orig_i of the mapped coordinate i is given by:
      1) i < destination: orig_i = slope_0 * i.
      2) destination <= i < choose_range:
         orig_i = slope_1 * i - (slope_1 - slope_0) * destination.
      3) i >= choose_range: orig_i = i.

    Denoting n_i = ceil(orig_i), the warp matrix element warp[i][j] is given by:
      1) j = n_i: 1 - n_i + orig_i.
      2) j = n_i - 1: n_i - orig_i.
      3) Otherwise: 0.

    Applying the warp matrix to an array of pixels, i.e.,
    warped_pixel[i] = sum_j warp[i][j] * pixel[j], one would get
    warped_pixel[i] = (n_i-orig_i) pixel[n_i-1] + (1-n_i+orig_i) pixel[n_i].

    Args:
      batch_size: Batch size. Integer number.
      matrix_size: Dimension of the vector space the warp matrix is applied to.
        Integer number.
      origin: Origin anchor point for warping. Tensor of shape (batch_size,) and
        data type dtype.
      destination: Destination of the origin anchor point upon warping. Tensor
        of shape (batch_size,) and data type dtype.
      choose_range: Range within which the warp reference points must lie.
        Tensor of shape (batch_size,) data type dtype.
      dtype: Data type of origin, destination, choose_range and the output warp
        matrix.

    Returns:
      warp_matrix: An array of fixed size warp matrices with shape
      (batch_size, matrix_size, matrix_size).
    """
    p = self.params

    # Entries of destination must be in the range
    # 1 <= destination <= choose_range - 1
    # for warp matrix to have non-singular values.
    destination = tf.minimum(tf.maximum(destination, 1.0), choose_range - 1.0)

    # Construct piece-wise linear function fixing boundary points
    # specified by zero, choose_range and matrix size and maps
    # the origin anchor point to the destination.
    destination_bc = tf.broadcast_to(destination, (matrix_size, batch_size))
    destination_bc = tf.transpose(destination_bc)
    choose_range_bc = tf.broadcast_to(choose_range, (matrix_size, batch_size))
    choose_range_bc = tf.transpose(choose_range_bc)

    # Slopes of piece-wise linear function.
    slope_0 = origin / destination
    slope_1 = (choose_range - origin) / (choose_range - destination)
    slope_2 = 1.0

    # x is a batch of origin matrices.
    # The origin matrix is the matrix such that
    # origin[i][j] = Origin coordinate of coordinate i for the warp map.
    # Denoting the destination of the origin anchor point in the
    # warp map as "dest," the origin coordinate of point i is given by:
    # 1) i < dest: slope_0 * i.
    # 2) dest <= i < choose_range: slope_1 * i - (slope_1 - slope_0) * dest.
    # 3) i >= choose_range: i.
    x = tf.broadcast_to(
        tf.cast(tf.range(matrix_size), dtype=dtype), (batch_size, matrix_size))
    x = (
        self.EinsumBBmBm(slope_0, x) +
        self.EinsumBBmBm(slope_1 - slope_0, tf.nn.relu(x - destination_bc)) +
        self.EinsumBBmBm(slope_2 - slope_1, tf.nn.relu(x - choose_range_bc)))
    x = tf.broadcast_to(x, (matrix_size, batch_size, matrix_size))
    x = tf.transpose(x, perm=[1, 2, 0])

    # y is a batch of coordinate matrices.
    # A coordinate matrix is a matrix such that
    # coordinate[i][j] = j.
    y = tf.broadcast_to(
        tf.cast(tf.range(matrix_size), dtype=dtype),
        (batch_size, matrix_size, matrix_size))
    # Warp matrix is obtained by applying hat function element-wise to (x-y).
    # Denoting the origin point of i under the warp map as orig_i,
    # and n_i = ceil(orig_i), the warp matrix element warp[i][j] is given by:
    # 1) j = n_i: 1 - n_i + orig_i.
    # 2) j = n_i - 1: n_i - orig_i.
    # 3) Otherwise: 0.
    # Applying the warp matrix to pixels, i.e.,
    # warped_pixel[i] = sum_j warp[i][j] * original_pixel[j], one would get
    # warped_pixel[i] = (n_i - orig_i) * original_pixel[n_i-1]
    #                   + (1 - n_i + orig_i) * original_pixel[n_i].
    warp_matrix = x - y
    warp_matrix = _hat(warp_matrix)
    if p.fprop_dtype is not None and p.fprop_dtype != dtype:
      warp_matrix = tf.cast(warp_matrix, p.fprop_dtype)

    return warp_matrix

  def _FrequencyMask(self,
                     inputs,
                     global_seed,
                     dtype=tf.float32,
                     domain_id_index=0):
    """Applies frequency masking with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      global_seed: an integer seed tensor for stateless random ops.
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
        global_seed=global_seed,
        max_length=freq_mask_max_bins,
        masks_per_frame=0.0,
        multiplicity=multiplicity,
        dtype=dtype,
        max_ratio=1.0)
    return self.EinsumBxycByBxyc(inputs, block_arrays)

  def _TimeMask(self,
                inputs,
                seq_lengths,
                global_seed,
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
      global_seed: an integer seed tensor for stateless random ops.
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
        global_seed=global_seed,
        max_length=time_mask_max_frames,
        masks_per_frame=time_masks_per_frame,
        multiplicity=multiplicity,
        dtype=dtype,
        max_ratio=max_ratio)

    # Non-empty random seed values are only used for testing or when using
    # stateless random ops. seed_6 and seed_7 are set separately to avoid
    # correlation of warp magnitude and origin position.
    if p.use_input_dependent_random_seed:
      seed_6 = global_seed + 6
      seed_7 = global_seed + 7
    else:
      seed_6 = p.random_seed
      seed_7 = p.random_seed

    outputs = self.EinsumBxycBxBxyc(
        inputs, block_arrays, name='einsum_formasking')
    if noisify:
      # Sample noise with standard deviation with factor * 0.1 + 0.0001
      # TODO(ngyuzh): Make sure this won't affect EOS.
      if gaussian_noise:
        stddev = 1.0
      else:
        random_uniform = _random_uniform_op(p.use_input_dependent_random_seed)
        factor = random_uniform(
            shape=(), minval=1.0, maxval=2.0, dtype=dtype, seed=seed_6)
        stddev = factor * 0.1 + 0.0001
      random_normal = _random_normal_op(p.use_input_dependent_random_seed)
      noise = random_normal(
          shape=[tf.shape(inputs)[0],
                 tf.shape(inputs)[1],
                 tf.shape(inputs)[2]],
          stddev=stddev,
          seed=seed_7)
      if p.fprop_dtype is not None and p.fprop_dtype != p.dtype:
        noise = tf.cast(noise, p.fprop_dtype)
      outputs_mask = self.EinsumBxyBxBxy(
          noise, 1.0 - block_arrays, name='einsum_fornoisymasking')
      outputs = outputs + tf.expand_dims(outputs_mask, -1)

    return outputs

  def _FrequencyWarp(self,
                     inputs,
                     global_seed,
                     dtype=tf.float32,
                     domain_id_index=0):
    """Applies frequency warping with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      global_seed: an integer seed tensor for stateless random ops.
      dtype: Data type.
      domain_id_index: Domain ID index.

    Returns:
      Inputs with random frequency warping applied.
    """
    p = self.params
    batch_size, _, num_freq, _ = py_utils.GetShape(inputs)

    # Get parameters for warping.
    freq_warp_max_bins = p.freq_warp_max_bins[domain_id_index]

    # If maximum warp length is zero, do nothing.
    if freq_warp_max_bins == 0:
      return inputs
    choose_range = tf.ones((batch_size,), dtype=tf.int32) * num_freq

    # Create warping matrix in time direction and apply
    warp_matrix = self._GetWarpMatrix(
        batch_size,
        choose_range=choose_range,
        matrix_size=num_freq,
        global_seed=global_seed,
        max_warp_frames=freq_warp_max_bins,
        dtype=dtype)

    return self.EinsumBxycBzyBxzc(
        inputs, warp_matrix, name='einsum_forfreqwarping')

  def _TimeWarp(self,
                inputs,
                seq_lengths,
                global_seed,
                dtype=tf.float32,
                domain_id_index=0):
    """Applies time warping with given degree to inputs.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      seq_lengths: The actual sequence lengths which mask been sampled of shape
        (batch_size,).
      global_seed: an integer seed tensor for stateless random ops.
      dtype: Data type.
      domain_id_index: Domain ID index.

    Returns:
      Inputs with random time warping applied.
    """
    p = self.params
    batch_size, time_length, _, _ = py_utils.GetShape(inputs)

    # Get parameters for warping.
    time_warp_max_frames = p.time_warp_max_frames[domain_id_index]
    max_ratio = p.time_warp_max_ratio[domain_id_index]
    time_warp_bound = p.time_warp_bound[domain_id_index]
    assert time_warp_bound in ('static', 'dynamic')

    # If maximum warp length is zero, do nothing.
    if ((time_warp_max_frames == 0 and time_warp_bound == 'static') or
        max_ratio <= 0.0):
      return inputs
    seq_lengths = tf.cast(seq_lengths, tf.int32)

    # Discard upper-bound on time-warp frames when
    # dynamic time warping is used.
    if time_warp_bound == 'dynamic':
      time_warp_max_frames = None

    # Create warping matrix in time direction and apply
    warp_matrix = self._GetWarpMatrix(
        batch_size,
        choose_range=seq_lengths,
        matrix_size=time_length,
        global_seed=global_seed,
        max_warp_frames=time_warp_max_frames,
        dtype=dtype,
        max_ratio=max_ratio)

    return self.EinsumBxycBzxBzyc(inputs, warp_matrix, name='einsum_forwarping')

  def _FrequencyNoise(self,
                      inputs,
                      global_seed,
                      dtype=tf.float32,
                      domain_id_index=0):
    """Applies frequency noise with given degree to inputs.

    It samples N(1, stddev) in frequency space. e.g. stddev = 0.1
    It multiplies the noise scale vector to logmel features.

    Rationale of this augmentation:
      1) Multiplication in freq space is convolution in time space.
      2) White noise has a periodical pattern in time space, which can be
         represented by convolution operation [1][2].
      3) So multiplication in freq space imitates white noise.

    [1] https://ccrma.stanford.edu/~jos/sasp/Filtered_White_Noise.html
    [2] https://en.wikipedia.org/wiki/White_noise#Discrete-time_white_noise

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      global_seed: an integer seed tensor for stateless random ops.
      dtype: Data type.
      domain_id_index: Domain ID index.

    Returns:
      Inputs with random frequency warping applied.
    """
    p = self.params
    batch_size, _, num_freq, _ = py_utils.GetShape(inputs)

    # Get parameters for noise.
    freq_noise_max_stddev = p.freq_noise_max_stddev[domain_id_index]

    # If maximum noise stddev is zero, do nothing.
    if freq_noise_max_stddev <= 0.0:
      return inputs

    freq_noise_max_stddev = freq_noise_max_stddev * self.augment_weight

    # Non-empty random seed values are only used for testing or when using
    # stateless random ops. seed_1 and seed_2 are set separately to avoid
    # correlation of random_uniform and random_normal.
    if p.use_input_dependent_random_seed:
      seed_1 = global_seed + 1
      seed_2 = global_seed + 2
    elif p.random_seed:
      seed_1 = p.random_seed + 1
      seed_2 = 2 * p.random_seed
    else:
      seed_1 = p.random_seed
      seed_2 = p.random_seed
    random_uniform = _random_uniform_op(p.use_input_dependent_random_seed)
    random_normal = _random_normal_op(p.use_input_dependent_random_seed)

    stddev = random_uniform(
        shape=[batch_size, 1, 1, 1],
        minval=0,
        maxval=freq_noise_max_stddev,
        seed=seed_1)
    noise_scale = random_normal(
        shape=[batch_size, 1, num_freq, 1], mean=1., stddev=stddev, seed=seed_2)
    return inputs * noise_scale

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
                           inputs,
                           paddings,
                           global_seed,
                           domain_id_index=0):
    """Returns augmented features.

    Args:
      inputs: Batch of input features of shape (batch_size, time_length,
        num_freq, channels).
      paddings: Batch of padding vectors of shape (batch_size, time_length).
      global_seed: an integer seed tensor for stateless random ops.
      domain_id_index: domain id index.

    Returns:
      Batch of output features of shape (batch_size, time_length, num_freq,
      channels) obtained by applying random augmentations to inputs.
    """
    p = self.params
    dtype = p.dtype

    # Unstack the features.
    if p.unstack:
      original_shape = tf.shape(inputs)
      inputs, paddings = self.UnstackFeatures(inputs, paddings)

    lengths = tf.reduce_sum(1 - paddings, 1)
    inputs = self._FrequencyWarp(
        inputs,
        global_seed=global_seed,
        dtype=dtype,
        domain_id_index=domain_id_index)
    inputs = self._TimeWarp(
        inputs,
        lengths,
        global_seed=global_seed,
        dtype=dtype,
        domain_id_index=domain_id_index)
    inputs = self._FrequencyNoise(
        inputs,
        global_seed=global_seed,
        dtype=dtype,
        domain_id_index=domain_id_index)
    inputs = self._TimeMask(
        inputs,
        lengths,
        global_seed=global_seed,
        noisify=p.use_noise,
        gaussian_noise=p.gaussian_noise,
        dtype=dtype,
        domain_id_index=domain_id_index)
    inputs = self._FrequencyMask(
        inputs,
        global_seed=global_seed,
        dtype=dtype,
        domain_id_index=domain_id_index)

    # Restack the features after applying specaugment.
    if p.unstack:
      inputs = tf.reshape(inputs, original_shape)

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

    global_seed = None  # A tensor seed in case stateless random ops are needed.
    if p.use_input_dependent_random_seed:
      global_seed = _global_seed_from_inputs(inputs)

    batch_size = py_utils.GetShape(inputs)[0]
    if len(p.domain_ids) > 1:
      augmented_inputs = tf.zeros_like(inputs)
      original_inputs = inputs
      for i, domain_id in enumerate(p.domain_ids):
        augmented_domain = self._AugmentationNetwork(
            inputs,
            paddings,
            global_seed=global_seed,
            domain_id_index=i)
        target_domain = tf.cast(
            tf.expand_dims(tf.tile([domain_id], [batch_size]), -1),
            dtype=p.dtype)
        # [batch, time].
        domain_mask = tf.cast(
            tf.equal(domain_ids, target_domain), dtype=p.dtype)
        augmented_domain = self.EinsumBxycBxBxyc(
            augmented_domain, domain_mask, name='einsum_domainmasking')
        original_inputs = self.EinsumBxycBxBxyc(
            original_inputs, 1.0 - domain_mask, name='einsum_domainmasking2')
        augmented_inputs = augmented_domain + augmented_inputs
      augmented_inputs = original_inputs + augmented_inputs
    else:
      augmented_inputs = self._AugmentationNetwork(
          inputs,
          paddings,
          global_seed=global_seed,
          domain_id_index=0)
    return augmented_inputs, paddings
