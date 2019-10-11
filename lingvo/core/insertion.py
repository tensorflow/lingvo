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
"""Insertion-based Framework.

References:
  KERMIT: https://arxiv.org/pdf/1906.01604.pdf
  Insertion Transformer: https://arxiv.org/pdf/1902.03249.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils


def SequenceTrimLastToken(x, x_paddings):
  """Trims the last token off of sequence `x`, and set trimmed elements to 0.

  Args:
    x: A sequence of tokens of shape [batch_size, x_len_max].
    x_paddings: The paddings of `x`.

  Returns:
    A tuple.
      - The new sequence, Tensor of shape [batch_size, x_len_max].
      - The new paddings, Tensor of shape [batch_size, x_len_max].
  """
  x_len = tf.reduce_sum(1 - x_paddings, 1)
  x_len_max = py_utils.GetShape(x)[1]
  x_trimmed_len = tf.maximum(x_len - 1, 0)
  x_trimmed_paddings = tf.sequence_mask(x_trimmed_len, x_len_max,
                                        x_paddings.dtype)
  x_trimmed = x * tf.cast(x_trimmed_paddings, x.dtype)
  return x_trimmed, 1 - x_trimmed_paddings


def SequenceAppendToken(x, x_paddings, token, extend=False):
  """Appends <token> to sequence `x`.

  Args:
    x: A sequence of tokens of shape [batch_size, x_len_max].
    x_paddings: The paddings of `x`.
    token: The token to append (of type integer).
    extend: Whether to extend `x` along the length dimension, this must be true
      for any sequence length in `x` that is `x_len_max` or else an invalid
      sequence will be emitted.

  Returns:
    A tuple.
      - The new sequence, Tensor of shape [batch_size, x_len_max].
      - The new paddings, Tensor of shape [batch_size, x_len_max].
  """
  batch_size = py_utils.GetShape(x)[0]
  x_len = tf.cast(tf.round(tf.reduce_sum(1 - x_paddings, 1)), tf.int32)
  if extend:
    x = tf.pad(x, [[0, 0], [0, 1]])
  # Mask all invalid entries of `x` to 0.
  x *= tf.sequence_mask(x_len, py_utils.GetShape(x)[1], x.dtype)
  # Append the <token> based on `x_len`.
  x += tf.scatter_nd(
      tf.stack([tf.range(batch_size), x_len], axis=1),
      tf.cast(tf.fill([batch_size], token), x.dtype), py_utils.GetShape(x))
  x_paddings = 1 - tf.sequence_mask(x_len + 1,
                                    py_utils.GetShape(x)[1], x_paddings.dtype)
  return x, x_paddings


def SequenceConcat(x, x_paddings, y, y_paddings, pad=0):
  """Concats sequence `x` with sequence `y`.

  This function is length aware (based off the paddings).

  Args:
    x: A sequence of tokens of shape [batch_size, x_len_max].
    x_paddings: The paddings of `x`.
    y: A sequence of tokens of shape [batch_size, y_len_max].
    y_paddings: The paddings of `y`.
    pad: The <pad> token to fill the concatenated sequence (of type integer).

  Returns:
    A tuple.
      - Concatenation of `x` and `y` of shape
        [batch_size, x_len_max + y_len_max].
      - Paddings of the concatenation of shape
        [batch_size, x_len_max + y_len_max].
  """
  # Get the length (w/ eos).
  x_len = tf.cast(tf.round(tf.reduce_sum(1 - x_paddings, 1)), tf.int32)
  y_len = tf.cast(tf.round(tf.reduce_sum(1 - y_paddings, 1)), tf.int32)

  batch_size = py_utils.GetShape(x)[0]
  y_len_max = py_utils.GetShape(y)[1]

  # Pad `x` with necessary <pad>.
  x = tf.concat([x, tf.fill(py_utils.GetShape(y), pad)], 1)
  # Replace all <pad> with 0.
  x = tf.where(tf.not_equal(x, pad), x, tf.fill(py_utils.GetShape(x), 0))

  # Compute the write indices of `y` in `xy`.
  indices = tf.stack([
      tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, y_len_max]),
      (tf.tile(tf.expand_dims(tf.range(y_len_max), 0), [batch_size, 1]) +
       tf.expand_dims(x_len, 1)),
  ], 2)

  xy = x + tf.scatter_nd(indices, y, py_utils.GetShape(x))

  # We need to remap all <pad> to `pad`.
  xy = tf.where(
      tf.less(
          tf.expand_dims(tf.range(py_utils.GetShape(xy)[1]), 0),
          tf.expand_dims(x_len + y_len, 1)), xy,
      tf.fill(py_utils.GetShape(xy), pad))
  xy_paddings = 1 - tf.sequence_mask(x_len + y_len,
                                     py_utils.GetShape(xy)[1], x_paddings.dtype)
  return xy, xy_paddings


class SymbolInsertionLayer(base_layer.BaseLayer):
  """Insertion-based framework for symbols.

  This constructs the sampled rollin (observed) canvas, as well as the targets
  for an insertion-based model.
  """

  @classmethod
  def Params(cls):
    p = super(SymbolInsertionLayer, cls).Params()
    p.Define(
        'rollin_policy', 'oracle',
        'Rollin policy, should be {oracle, uniform}. Rollin policy is the '
        'sampling policy from which we draw the canvas. '
        '`oracle` means same as `oracle_policy`.')
    p.Define(
        'oracle_policy', 'uniform',
        'Oracle policy, should be one of {uniform}. Oracle policy is the '
        'target policy from which we select our targets and train our '
        'models.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SymbolInsertionLayer, self).__init__(params)

  def FProp(self,
            theta,
            x,
            x_paddings=None,
            eos_id=1,
            force_sample_last_token=True):
    """Applies SymbolInsertionLayer.

    We take in a `x`, which represents the groundtruth sequence (i.e., English
    sequence). We return a sampled rollin (observed) canvas (i.e., random subset
    of the English sequence), as well as the target (indices) for an
    insertion-based model (i.e., the targets given the random observed subset).

    Args:
      theta: Ignored, this can be None.
      x: The symbol ids of shape `[batch_size, time_dim]`.
      x_paddings: The paddings (1 or 0) of shape `[batch_size, time_dim]` where
        0 is valid and 1 is invalid.
      eos_id: The <eos> token id to represent end-of-slot.
      force_sample_last_token: Set True to force sample the last token of `x`.

    Returns:
      A `NestedMap`.
        - canvas: The canvas (based off of the `rollin_policy`) of shape
          [batch_size, c_dim]. Note that, `c_dim` <= `time_dim` but need not be
          equal.
        - canvas_indices: The canvas indices (into `x`).
        - canvas_paddings: The paddings of `canvas_indices`.
        - target_indices: The target indices of shape [num_targets, 3].
          `num_targets` is the number of total targets in the entire batch.
          [:, 0] captures the batch, [:, 1] captures the slot, and [:, 2]
          captures the token. Each row [batch, slot, vocab] represents the
          indices of the target -- i.e., the batch, slot and vocab combination
          of the target. Typical usage of these indices is to tf.gather_nd
          the log-probs (from the softmax layer).
        - target_weights: The target weights.

    Raises:
      ValueError: If invalid params.
    """
    p = self.params

    batch_size = py_utils.GetShape(x)[0]
    time_dim = py_utils.GetShape(x)[1]

    if x_paddings is None:
      x_paddings = tf.zeros([batch_size, time_dim], tf.float32)

    oracle_policy = p.oracle_policy
    rollin_policy = (
        oracle_policy if p.rollin_policy == 'oracle' else p.rollin_policy)

    if rollin_policy != 'uniform':
      raise ValueError('Unknown or unsupported rollin policy: %s' %
                       rollin_policy)
    if oracle_policy != 'uniform':
      raise ValueError('Unknown or unsupported oracle policy: %s' %
                       oracle_policy)

    x_len = tf.cast(tf.round(tf.reduce_sum(1 - x_paddings, 1)), tf.int32)

    # Compute the desired length per example in the batch.
    ratio = tf.random.uniform([batch_size], 0.0, 1.0, seed=p.random_seed)
    if force_sample_last_token:
      c_len = tf.minimum(
          tf.cast(ratio * tf.cast(x_len, tf.float32), tf.int32), x_len - 1) + 1
    else:
      c_len = tf.minimum(
          tf.cast(ratio * tf.cast(x_len + 1, tf.float32), tf.int32), x_len)
    # Compute the maximum length across the batch.
    c_len_max = tf.reduce_max(c_len)

    # Grab subset of random valid indices per example.
    z_logits = tf.cast(
        tf.expand_dims(tf.range(time_dim), 0) >= tf.expand_dims(x_len, 1),
        tf.float32) * -1e9
    if force_sample_last_token:
      # Force sample the last token -- i.e., as indexed by `x_len - 1`. We can
      # accomplish this by add +LARGE_NUMBER to the logits.
      z_logits += tf.cast(
          tf.equal(
              tf.expand_dims(tf.range(time_dim), 0), tf.expand_dims(
                  x_len - 1, 1)), tf.float32) * 1e9
    # Gumbel-max trick to sample (we only sample valid positions per sample in
    # the batch).
    z = -tf.math.log(-tf.math.log(
        tf.random.uniform([batch_size, time_dim], seed=p.random_seed)))
    unused_c_values, c_indices = tf.nn.top_k(z_logits + z, time_dim)

    # Trim everything > c_len_max.
    c_indices = c_indices[:, :c_len_max]

    # Invalidate any indices >= c_len, we use the last index as the default
    # invalid index.
    c_indices = tf.where(
        tf.expand_dims(tf.range(c_len_max), 0) < tf.expand_dims(c_len, 1),
        c_indices, tf.fill(py_utils.GetShape(c_indices), time_dim - 1))

    # Materialize the canvas.
    c_indices = tf.sort(c_indices)
    c = tf.gather_nd(
        x,
        tf.stack([
            tf.reshape(
                tf.tile(
                    tf.expand_dims(tf.range(batch_size), 1), [1, c_len_max]),
                [-1]),
            tf.reshape(c_indices, [-1])
        ], 1))
    c = tf.reshape(c, [batch_size, c_len_max])

    # Compute the paddings.
    c_paddings = 1 - tf.sequence_mask(c_len, c_len_max, dtype=x_paddings.dtype)
    c *= tf.cast(1 - c_paddings, tf.int32)

    indices = tf.concat([
        tf.reshape(
            tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, c_len_max]),
            [batch_size * c_len_max, 1]),
        tf.reshape(c_indices, [batch_size * c_len_max, 1])
    ], 1)
    x_token_is_observed = tf.scatter_nd(
        indices, tf.ones([batch_size * c_len_max], tf.int32),
        py_utils.GetShape(x))
    # `x_segments` captures which slot each `x` belongs to (both observed and
    # tokens that need to be observed).
    x_segments = tf.cumsum(x_token_is_observed, 1, exclusive=True)

    x_token_is_observed = tf.cast(x_token_is_observed, tf.bool)
    prev_x_token_is_observed = tf.pad(
        x_token_is_observed[:, :-1], [[0, 0], [1, 0]], constant_values=True)
    x_token_is_observed = tf.reshape(x_token_is_observed, [-1])
    prev_x_token_is_observed = tf.reshape(prev_x_token_is_observed, [-1])
    x_is_valid = tf.cast(1 - x_paddings, tf.bool)
    x_is_valid = tf.reshape(x_is_valid, [-1])

    # Remap all the observed to <eos>, note some of these need a zero weight
    # (or else there would be <eos> and valid token in the same slot).
    target_indices = tf.cast(tf.reshape(x, [-1, 1]), tf.int32)
    target_indices = tf.where(
        x_token_is_observed, tf.fill(py_utils.GetShape(target_indices), eos_id),
        target_indices)

    # TODO(williamchan): We give uniform 1.0 weight, however, math suggests
    # we may want to weigh this term by the original sequence length.
    target_weights = tf.ones_like(target_indices, tf.float32)

    # We need to set all the weights for <eos> which actually have valid tokens
    # in the slot to zero.
    target_weights = tf.where(x_token_is_observed & ~prev_x_token_is_observed,
                              tf.zeros_like(target_weights), target_weights)

    # TODO(williamchan): Consider dropping the entries w/ weight zero.

    # Add the batch and slot indices.
    target_indices = tf.concat([
        tf.reshape(
            tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, time_dim]),
            [batch_size * time_dim, 1]),
        tf.reshape(x_segments, [-1, 1]), target_indices
    ], 1)

    # Select only the valid indices. The selected valid ones include slots w/
    # <eos>.
    target_indices = target_indices[x_is_valid]
    target_weights = target_weights[x_is_valid]

    return py_utils.NestedMap(
        canvas=c,
        canvas_indices=c_indices,
        canvas_paddings=c_paddings,
        target_indices=target_indices,
        target_weights=target_weights)
