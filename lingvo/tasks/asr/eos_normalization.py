# Lint as: python3
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
"""util functions for eos normalization."""
from typing import Tuple

from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np


def FillPaddingPos(ids: tf.Tensor, id_len: tf.Tensor,
                   padding_value: int) -> tf.Tensor:
  """Given a batch of sequences, fills the padding pos with `padding_value`.

  Args:
    ids: a [B, max_len] int tensor.
    id_len: a [B, ] int tensor.
    padding_value: an int.

  Returns:
    new_ids: new ids with the property.
      - new_ids[b, :id_len[b]] = ids[b, :id_len[b]]
      - new_ids[b, id_len[b]:] = padding_value
  """
  mask = py_utils.SequencePaddings(id_len, maxlen=tf.shape(ids)[1])
  mask = tf.cast(mask, dtype=tf.bool)
  new_ids = tf.where(mask, tf.fill(tf.shape(ids), padding_value), ids)
  return new_ids


def NormalizeTrailingEos(ids: tf.Tensor,
                         id_len: tf.Tensor,
                         need_trailing_eos: bool = True,
                         eos_id: int = 2) -> Tuple[tf.Tensor, tf.Tensor]:
  """Optionally removes/adds the trailing eos symbol.

  Given ids/id_len, return normalized id_len, and also make sure the padding
  positions are filled with eos.

  Specifically,
    - If need_trailing_eos = True and the last symbol is:
      * eos: id_len_eos_normalized = id_len
      * not eos: id_len_eos_normalized = min(id_len + 1, max_label_len)
    - If need_trailing_eos = False and the last symbol is:
      * eos: id_len_eos_normalized = max(id_len - 1, 0)
      * not eos: id_len_eos_normalized = id_len

  Args:
    ids: a [B, max_label_len] int tensor.
    id_len: a [B,] int tensor. `id_len` indicates the last symbol's position.
    need_trailing_eos: bool. if True, then the return id_len include the last
      eos symbol; otherwise, it does not include the last eos.
    eos_id: int. The index of eos symbol.

  Returns:
    new_ids: a [B, max_label_len] int tensor, and it is guaranteed that:
      * new_ids[b, :min(id_len_eos_normalized[b], id_len[b])] =
        ids[b, :min(id_len_eos_normalized[b], id_len[b])]
      * new_ids[b, id_len_eos_normalized[b]:] = eos_id.
    id_len_eos_normalized: a [B, ] int tensor, which indicates eos normalized
      length.
  """
  new_ids = FillPaddingPos(ids, id_len, padding_value=eos_id)
  batch_size, max_len = py_utils.GetShape(new_ids, 2)
  indices_x = tf.range(batch_size)
  indices_y = tf.maximum(id_len - 1, 0)
  indices = tf.concat([indices_x[:, tf.newaxis], indices_y[:, tf.newaxis]],
                      axis=-1)
  last_token = tf.gather_nd(new_ids, indices)
  last_token_is_eos = tf.equal(last_token, eos_id)
  if need_trailing_eos:
    id_len_eos_normalized = tf.where(last_token_is_eos, id_len, id_len + 1)
    # because we did id_len+1, it is possible that the id_len_eos_normalized
    # is larger than max_label_len, so we need to cap id_len_eos_normalized
    id_len_eos_normalized = tf.minimum(id_len_eos_normalized, max_len)
  else:
    id_len_eos_normalized = tf.where(last_token_is_eos, id_len - 1, id_len)
    id_len_eos_normalized = tf.maximum(id_len_eos_normalized, 0)
  return new_ids, id_len_eos_normalized


def NumpyNormalizeTrailingEos(ids: np.ndarray,
                              id_len: np.ndarray,
                              need_trailing_eos: bool = True,
                              eos_id: int = 2) -> Tuple[np.ndarray, np.ndarray]:
  """Optionally removes/adds the trailing eos symbol, numpy implementation.

  This is the numpy implementation of `NormalizeTrailingEos`. See more details
  there. As only a reference implementation, it is not optimized but it should
  sever better for readers to understand the logic and for debug purpose
  as well.

  Args:
    ids: a [B, max_label_len] int np.array.
    id_len: a [B,] int np.array. `id_len` indicates the last symbol's position.
    need_trailing_eos: bool. if True, then the return id_len include the last
      eos symbol; otherwise, it does not include the last eos.
    eos_id: int. The index of eos symbol.

  Returns:
    new_ids: a [B, max_label_len] np.array, and it is guaranteed that:
      * new_ids[b, :min(id_len_eos_normalized[b], id_len[b])] =
        ids[b, :min(id_len_eos_normalized[b], id_len[b])]
      * new_ids[b, id_len_eos_normalized[b]:] = eos_id.
    id_len_eos_normalized: a [B, ] np.array, which indicates eos normalized
      length.
  """
  new_ids = np.zeros_like(ids)
  id_len_eos_normalized = np.zeros_like(id_len)
  (batch_size, max_label_len) = ids.shape

  def CopyToNewIds(ids, new_ids, seq_num, end_pos):
    new_ids[seq_num, :end_pos] = ids[seq_num, :end_pos]

  def PadNewIdWithEos(new_ids, seq_num, start_pos):
    new_ids[seq_num, start_pos:] = eos_id

  for b in range(batch_size):
    if ids[b, id_len[b] - 1] == eos_id:
      if need_trailing_eos:
        id_len_eos_normalized[b] = id_len[b]
      else:
        id_len_eos_normalized[b] = max(id_len[b] - 1, 0)
      CopyToNewIds(ids, new_ids, b, id_len_eos_normalized[b])
    else:
      if need_trailing_eos:
        id_len_eos_normalized[b] = min(max_label_len, id_len[b] + 1)
        new_ids[b, id_len_eos_normalized[b] - 1] = eos_id
      else:
        id_len_eos_normalized[b] = id_len[b]
      CopyToNewIds(ids, new_ids, b, id_len[b])

    PadNewIdWithEos(new_ids, b, id_len_eos_normalized[b])

  return new_ids, id_len_eos_normalized
