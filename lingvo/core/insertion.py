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
  Insertion Transformer: https://arxiv.org/pdf/1902.03249.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from lingvo.core import base_layer
from lingvo.core import py_utils


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

  def FProp(self, theta, x, x_paddings=None):
    """Apply SymbolInsertionLayer.

    We take in a `x`, which represents the groundtruth sequence (i.e., English
    sequence). We return a sampled rollin (observed) canvas (i.e., random subset
    of the English sequence), as well as the target (indices) for an
    insertion-based model (i.e., the targets given the random observed subset).

    Args:
      theta: This should be None.
      x: The symbol ids of shape `[batch_size, time_dim]`.
      x_paddings: The paddings (1 or 0) of shape `[batch_size, time_dim]` where
        0 is valid and 1 is invalid.

    Returns:
      A `NestedMap`.
        - canvas: The canvas (based off of the `rollin_policy`) of shape
          [batch_size, c_dim].
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

    batch_size, time_dim = py_utils.GetShape(x)

    if x_paddings is None:
      x_paddings = tf.zeros([batch_size, time_dim], x.dtype)

    oracle_policy = p.oracle_policy
    rollin_policy = (
        oracle_policy if p.rollin_policy == 'oracle' else p.rollin_policy)

    if rollin_policy != 'uniform':
      raise ValueError('Unknown or unsupported rollin policy: %s' %
                       rollin_policy)
    if oracle_policy != 'uniform':
      raise ValueError('Unknown or unsupported oracle policy: %s' %
                       oracle_policy)

    x_len = tf.reduce_sum(1 - x_paddings, 1)

    # Compute the desired length per example in the batch.
    ratio = tf.random.uniform([batch_size], 0.0, 1.0)
    c_len = tf.minimum(tf.cast(ratio * tf.to_float(x_len + 1), tf.int32), x_len)
    # Compute the maximum length across the batch.
    c_len_max = tf.reduce_max(c_len)

    # Grab subset of random valid indices per example.
    z_logits = tf.cast(
        tf.expand_dims(tf.range(time_dim), 0) >= tf.expand_dims(x_len, 1),
        tf.float32) * -1e9
    # Gumbel-max trick to sample (we only sample valid positions per sample in
    # the batch).
    z = -tf.math.log(-tf.math.log(tf.random.uniform([batch_size, time_dim])))
    unused_c_values, c_indices = tf.nn.top_k(z_logits + z, time_dim, True)

    # Trim everything > c_len_max.
    c_indices = c_indices[:, :c_len_max]

    # Invalidate any indices >= c_len.
    c_indices = tf.where(
        tf.expand_dims(tf.range(c_len_max), 0) < tf.expand_dims(c_len, 1),
        c_indices, tf.fill(py_utils.GetShape(c_indices), time_dim - 1))

    # Materialize the canvas.
    c_indices = tf.sort(c_indices)
    c = tf.reshape(
        tf.gather(tf.reshape(x, [-1]), tf.reshape(c_indices, [-1])),
        [batch_size, c_len_max])

    # Compute the paddings.
    c_paddings = 1 - tf.sequence_mask(c_len, c_len_max, dtype=c.dtype)

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
    x_segments = tf.cumsum(x_token_is_observed, 1)

    # The target indices (to tf.gather_nd the log-probs).
    target_indices = tf.concat([
        tf.reshape(
            tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, time_dim]),
            [batch_size * time_dim, 1]),
        tf.reshape(x_segments, [-1, 1]),
        tf.reshape(x, [-1, 1])
    ], 1)
    # TODO(williamchan): We give uniform 1.0 weight, however, math suggests
    # we may want to weigh this term by the original sequence length.
    target_weights = tf.ones([batch_size * time_dim], tf.float32)

    x_token_is_observed = tf.reshape(
        tf.cast(x_token_is_observed, tf.bool), [-1])
    x_is_valid = tf.reshape(tf.cast(1 - x_paddings, tf.bool), [-1])
    target_indices = target_indices[~x_token_is_observed & x_is_valid]
    target_weights = target_weights[~x_token_is_observed & x_is_valid]

    return py_utils.NestedMap(
        canvas=c,
        canvas_indices=c_indices,
        canvas_paddings=c_paddings,
        target_indices=target_indices,
        target_weights=target_weights)
