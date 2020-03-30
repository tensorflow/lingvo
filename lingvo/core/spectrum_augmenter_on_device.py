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
"""Lingvo layers that are used for spectrum augmentation on-device."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lingvo.compat as tf
from lingvo.core import spectrum_augmenter


class SpectrumAugmenterOnDevice(spectrum_augmenter.SpectrumAugmenter):
  """Performs data augmentation as according to the SpecAug paper.

  This implementation uses portable replacements for the tf.einsum ops.

  https://arxiv.org/pdf/1904.08779.pdf
  """

  def EinsumBBmBm(self, a, b, name=None):
    """Portable replacement for tf.einsum('b,bm->bm', a, b)."""
    return tf.math.multiply(tf.expand_dims(a, axis=-1), b, name=name)

  def EinsumBmtBmBt(self, a, b, name=None):
    """Portable replacement for tf.einsum('bmt,bm->bt', a, b)."""
    return tf.linalg.matvec(a, b, transpose_a=True, name=name)

  def EinsumBxycByBxyc(self, a, b, name=None):
    """Portable replacement for tf.einsum('bxyc,by->bxyc', a, b)."""
    expanded_b = tf.expand_dims(tf.expand_dims(b, axis=1), axis=3)
    return tf.math.multiply(a, expanded_b, name=name)

  def EinsumBxycBxBxyc(self, a, b, name=None):
    """Portable replacement for tf.einsum('bxyc,bx->bxyc', a, b)."""
    expanded_b = tf.expand_dims(tf.expand_dims(b, axis=2), axis=3)
    return tf.math.multiply(a, expanded_b, name=name)

  def EinsumBxyBxBxy(self, a, b, name=None):
    """Portable replacement for tf.einsum('bxy,bx->bxy', a, b)."""
    return tf.math.multiply(a, tf.expand_dims(b, axis=2), name=name)

  def EinsumBxycBzxBzyc(self, a, b, name=None):
    """Portable replacement for tf.einsum('bxyc,bzx->bzyc', a, b)."""
    expanded_a = tf.expand_dims(a, axis=1)
    expanded_b = tf.expand_dims(tf.expand_dims(b, axis=-1), axis=-1)
    return tf.reduce_sum(
        tf.math.multiply(expanded_a, expanded_b, name=name), axis=2)
