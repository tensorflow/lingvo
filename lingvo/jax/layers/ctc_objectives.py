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
"""Objective function for Connectionist Temporal Classification (CTC)."""

from typing import Mapping, Tuple

import jax
import jax.numpy as jnp
from lingvo.jax import asserts
from lingvo.jax import pytypes

JTensor = pytypes.JTensor
_LOGEPSILON = -100000.0


def ctc_loss(logits: JTensor,
             logitpaddings: JTensor,
             labels: JTensor,
             labelpaddings: JTensor,
             blank_id: int = 0) -> Tuple[JTensor, Mapping[str, JTensor]]:
  """Computes CTC loss.

  This function performs forward computation over an FSA with `N * 2` states
  where `N` is the max number of labels. The states are split into two groups:
  Phi states and emission states. a phi-state accepts repetition of
  phi (blank)-symbols and transits to emission state when the correct label is
  observed. An emission state accepts repetition of the label and transits to
  the next phi states at any time (so called epsilon-transition).
  Below, `B` denotes the batch size, `T` denotes the time steps in `logits`,
  and `N` denotes the time steps in `labels`.

  Args:
    logits: (B, T, K)-array containing log-probabilities of each class.
    logitpaddings: (B, T)-array. Padding indicators for `logits`.
    labels: (B, N)-array containing reference integer labels.
    labelpaddings: (B, N)-array. Padding indicators for `labels`. Currently,
      `labels` must be right-padded, i.e. each row of `labelpaddings` must be
      repetition of zeroes, followed by repetition of ones.
    blank_id: Id for blank token.

  Returns:
    A pair of `(per_seq_loss, aux)`.
    per_seq_loss:
      (B,)-array containing loss values for each sequence in the batch.
    aux: Dictionary containing interim variables used for computing losses.
      aux['logalpha_phi']: (T, B, N+1)-array. Log-forward-probabilities of each
        phi-state corresponding to the n-th label.
      aux['logalpha_emit']: (T, B, N)-array. Log-forward-probabilities of each
        emission-state corresponding to the n-th label.
      aux['logprobs_phi']: (T, B, 1)-array. Probability of the phi-symbol
        corresponding to each time frame.
      aux['logprobs_emit']: (T, B, N)-array. Probability of the n-th label
        corresponding to each time frame.
  """
  batchsize, unused_maxinputlen, num_classes = logits.shape
  batchsize_, maxlabellen = labels.shape
  asserts.eq(batchsize, batchsize_)

  logprobs = jax.nn.log_softmax(logits)
  labellens = maxlabellen - jnp.sum(labelpaddings, axis=1).astype(jnp.int32)

  # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
  repeat = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)
  repeat = jnp.pad(repeat, ((0, 0), (0, 1)))

  logprobs_phi = logprobs[:, :, blank_id:blank_id + 1]  # [B, T, 1]
  logprobs_phi = jnp.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

  one_hot = jax.nn.one_hot(labels, num_classes=num_classes)  # [B, N, K]
  logprobs_emit = jnp.einsum('btk,bnk->btn', logprobs, one_hot)
  logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

  logalpha_phi_init = jnp.ones(
      (batchsize, maxlabellen + 1)) * _LOGEPSILON  # [B, N]
  logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
  logalpha_emit_init = jnp.ones(
      (batchsize, maxlabellen)) * _LOGEPSILON  # [B, N]

  def loop_body(prev, x):
    prev_phi, prev_emit = prev
    # emit-to-phi epsilon transition, except if the next label is repetition
    prev_phi_orig = prev_phi
    prev_phi = prev_phi.at[:, 1:].set(
        jnp.logaddexp(prev_phi[:, 1:], prev_emit + _LOGEPSILON * repeat))

    logprob_emit, logprob_phi, pad = x

    # phi-to-emit transition
    next_emit = jnp.logaddexp(prev_phi[:, :-1] + logprob_emit,
                              prev_emit + logprob_emit)
    # self-loop transition
    next_phi = prev_phi + logprob_phi
    # emit-to-phi blank transition only when the next label is repetition
    next_phi = next_phi.at[:, 1:].set(
        jnp.logaddexp(next_phi[:, 1:],
                      prev_emit + logprob_phi + _LOGEPSILON * (1.0 - repeat)))

    pad = pad.reshape((batchsize, 1))
    next_emit = pad * prev_emit + (1.0 - pad) * next_emit
    next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

    return (next_phi, next_emit), (next_phi, next_emit)

  xs = (logprobs_emit, logprobs_phi, logitpaddings.transpose((1, 0)))
  _, (logalpha_phi,
      logalpha_emit) = jax.lax.scan(loop_body,
                                    (logalpha_phi_init, logalpha_emit_init), xs)

  # last row needs to be updated with the last epsilon transition
  logalpha_phi_last = logalpha_phi[-1].at[:, 1:].set(
      jnp.logaddexp(logalpha_phi[-1, :, 1:], logalpha_emit[-1]))
  logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

  # extract per_seq_loss
  one_hot = jax.nn.one_hot(labellens, num_classes=maxlabellen + 1)  # [B, N+1]
  per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)

  return per_seq_loss, {
      'logalpha_phi': logalpha_phi,
      'logalpha_emit': logalpha_emit,
      'logprobs_phi': logprobs_phi,
      'logprobs_emit': logprobs_emit,
  }


def sequence_mask(lengths: jnp.ndarray, maxlen):
  batch_size = lengths.shape[0]
  a = jnp.ones([batch_size, maxlen])
  b = jnp.cumsum(a, axis=-1)
  c = jnp.less_equal(b, lengths[:, jnp.newaxis]).astype(lengths.dtype)
  return c


def collapse_and_remove_blanks(labels: jnp.ndarray,
                               seq_length: jnp.ndarray,
                               blank_id: int = 0):
  """Merge repeated labels into single labels and remove the designated blank symbol.

  Args:
    labels: Array of shape (batch, seq_length)
    seq_length: Arrray of shape (batch), sequence length of each batch element.
    blank_id: Optional id of the blank symbol

  Returns:
    tuple of tf.SparseTensor of shape (batch, seq_length) with repeated labels
    collapsed, eg: [[A, A, B, B, A],
                    [A, B, C, D, E]] => [[A, B, A],
                                         [A, B, C, D, E]]
    and int tensor of shape [batch] with new sequence lengths.
  """
  b, t = labels.shape
  # Zap out blank
  blank_mask = 1 - jnp.equal(labels, blank_id)
  labels = (labels * blank_mask).astype(labels.dtype)

  # Mask labels that don't equal previous label.
  label_mask = jnp.concatenate([
      jnp.ones_like(labels[:, :1], dtype=jnp.int32),
      jnp.not_equal(labels[:, 1:], labels[:, :-1])
  ],
                               axis=1)

  # Filter labels that aren't in the original sequence.
  maxlen = labels.shape[1]
  seq_mask = sequence_mask(seq_length, maxlen=maxlen)
  label_mask = label_mask * seq_mask

  # remove repetitions from the labels
  ulabels = label_mask * labels

  # Count masks for new sequence lengths.
  label_mask = jnp.not_equal(ulabels, 0).astype(labels.dtype)
  new_seq_len = jnp.sum(label_mask, axis=1)

  # Mask indexes based on sequence length mask.
  new_maxlen = maxlen
  idx_mask = sequence_mask(new_seq_len, maxlen=new_maxlen)

  # Flatten everything and mask out labels to keep and sparse indices.
  flat_labels = jnp.reshape(ulabels, [-1])
  flat_idx_mask = jnp.reshape(idx_mask, [-1])

  indices = jnp.nonzero(flat_idx_mask, size=b * t)[0]
  values = jnp.nonzero(flat_labels, size=b * t)[0]
  updates = jnp.take_along_axis(flat_labels, values, axis=-1)

  # Scatter to flat shape.
  flat = jnp.zeros(flat_idx_mask.shape).astype(labels.dtype)
  flat = flat.at[indices].set(updates)
  # 0'th position in the flat array gets clobbered by later padded updates,
  # so reset it here to its original value
  flat = flat.at[0].set(updates[0])

  # Reshape back to square batch.
  batch_size = labels.shape[0]
  new_shape = [batch_size, new_maxlen]
  return (jnp.reshape(flat, new_shape).astype(labels.dtype),
          new_seq_len.astype(seq_length.dtype))
