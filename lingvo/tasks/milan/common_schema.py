# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Defines a common `tf.train.Example` format for image-caption(-like) data."""

import lingvo.compat as tf


def _Feature(shape, dtype=tf.float32):
  shape = tf.TensorShape(shape)
  if shape.is_fully_defined():
    return tf.io.FixedLenFeature(shape, dtype=dtype)
  else:
    if shape[:1].is_fully_defined() or not shape[1:].is_fully_defined():
      raise ValueError(f'Unsupported sequence shape {shape}')
    return tf.io.FixedLenSequenceFeature(
        shape[1:], dtype=dtype, allow_missing=True)


def ImageFeatures(images_per_example=1):
  """Returns definitions of common image features.

  Args:
    images_per_example: Number of images stored in each example.
  Returns:
    A dict of feature definitions usable with `tf.io.parse_example`.
  """
  images_shape = ([images_per_example] if images_per_example > 1 else [])

  return {
      'image/encoded': _Feature(images_shape, tf.string),
      'image/id': _Feature(images_shape, tf.int64),
  }


def TextFeatures(captions_per_example=1, bert_embeddings_shape=None):
  """Returns definitions of the common text features.

  Args:
    captions_per_example: Number of text captions stored in each example.
    bert_embeddings_shape: Optional time-major shape of BERT embedding sequences
      to include in the schema (if given). Set the leading (time) dimension to
      `None` if the sequences have variable length.

  Returns:
    A dict of feature definitions usable with `tf.io.parse_example`.
  """

  captions_shape = ([captions_per_example] if captions_per_example > 1 else [])

  features = {
      'text/captions': _Feature(captions_shape, tf.string),
      'text/id': _Feature(captions_shape, tf.int64),
  }

  if bert_embeddings_shape is not None:
    max_length, feature_dim = bert_embeddings_shape
    features.update({
        # Token-level BERT embeddings.
        'text/bert/token_features':
            _Feature(captions_shape + [max_length, feature_dim]),
        # Lengths (in tokens) of the token-level embeddings.
        'text/bert/lengths':
            _Feature(captions_shape, tf.int64)
    })
  return features


def AudioFeatures(mfcc_shape=None, cpc8k_shape=None):
  """Returns definitions of common audio features.

  Args:
    mfcc_shape: Optional time-major shape of MFCC features to include in the
      schema (if given). Set the leading (time) dimension to `None` if the
      sequences have variable length.
    cpc8k_shape: Optional time-major shape of CPC-8K features to include.

  Returns:
    A dict of feature definitions usable with `tf.io.parse_example`.
  """
  features = {'audio/id': _Feature([], tf.int64)}
  if mfcc_shape is not None:
    features['audio/mfccs'] = _Feature(mfcc_shape)
  if cpc8k_shape is not None:
    features['audio/cpc8k/features'] = _Feature(cpc8k_shape)
  return features
