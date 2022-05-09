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
"""Flat beam search helper for transformer decoder.

In flat beam search, all tokens in all hyps are packed into a single buffer of
size (beam_size * max_steps), and individual hyps are represented as 0/1 masks.

Decoder state (previously computed keys/values in decoder self-attention layers)
also has length (beam_size * max_steps).

Decoder is called for beam_size tokens at a time, thus decoder self-attention
is computed with query_size=beam_size and key_size=buf_size.

Decoder callback has signature:
  dec_callback(tgt_id, tgt_segment_id, tgt_pos, tgt_mask, dec_state, t)
      -> (logits, dec_state)
where
  tgt_id: int32 tensor of shape [batch_size, beam_size]
  tgt_semgent_id: int32 tensor of shape [batch_size, beam_size]
  tgt_pos: int32 tensor of shape [batch_size, beam_size]
  tgt_mask: float32 tensor of shape [batch_size, beam_size, beam_size*max_len]
  dec_state: decoder state
  t: t current step, int32 scalar

  logits: tensor of shape [batch_size, beam_size, vocab_size]
  dec_state: updated decoder state

The callback should update dec_state in range [t*beam_size:(t+1)*beam_size].
"""

from lingvo import compat as tf
from lingvo.core import tpu_summary


def einsum_i32(eq, *args):
  y = tf.einsum(eq, *[tf.cast(x, tf.int32) for x in args])
  return tf.cast(y, tf.int32)


def update_nbest(nbest_hyps, cur_hyps):
  """Updates nbest hyps from cur_hyps. Returns new values for nbest_hyps."""
  with tf.name_scope('update_nbest'):
    (nbest_mask, nbest_score, nbest_score_norm) = nbest_hyps
    (cur_mask, cur_score, cur_score_norm) = cur_hyps
    k = int(nbest_mask.shape[1])
    m = int(cur_mask.shape[1])
    mask = tf.concat([nbest_mask, cur_mask], 1)
    score = tf.concat([nbest_score, cur_score], 1)
    score_norm = tf.concat([nbest_score_norm, cur_score_norm], 1)
    nbest_score_norm, i = tf.math.top_k(score_norm, k)
    i_one_hot = tf.one_hot(i, k + m, dtype=mask.dtype)
    nbest_mask = tf.einsum('bkt,bjk->bjt', mask, i_one_hot)
    nbest_score = tf.einsum('bk,bjk->bj', score, i_one_hot)
    return (nbest_mask, nbest_score, nbest_score_norm)


def flat_beam_search(batch_size,
                     beam_size,
                     max_steps,
                     dec_callback,
                     dec_state,
                     bos_id=1,
                     eos_id=2,
                     length_norm_alpha=0.8,
                     beam_gap=3.0,
                     top_k_fn=tf.math.top_k,
                     prefix=None,
                     prefix_len=None,
                     fprop_dtype=tf.float32,
                     ext_size=0,
                     nbest_size=None,
                     debug=True):
  """Flat beam search.

  Args:
    batch_size: batch size
    beam_size: beam size limit in number of hyps
    max_steps: max steps
    dec_callback: decoder callback (see above)
    dec_state: decoder state
    bos_id: <s> token id
    eos_id: </s> token id
    length_norm_alpha: length normalization parameter
    beam_gap: early stopping threshold; None to disable
    top_k_fn: top_k function to call
    prefix: (optional) int32 tensor [batch_size, prefix_max]
    prefix_len: (optional) int32 tensor [batch_size]
    fprop_dtype: fprop dtype
    ext_size: int >= beam_size, extension buffer size
    nbest_size: number of returned hyps, default is beam_size
    debug: log intermediate vlaues with tpu_summary.tensor()

  Returns:
    (loop_vars, dec_state, nbest) where
    nbest = (topk_ids, topk_len, topk_score)
  """
  assert beam_size > 0
  assert batch_size > 0
  assert max_steps > 0

  buf_size = beam_size * max_steps
  output_len = max_steps

  if prefix is None:
    assert prefix_len is None
    # Create prefix of start tokens.
    prefix = tf.zeros([batch_size, beam_size], dtype=tf.int32)
    prefix += tf.one_hot(beam_size - 1, beam_size, dtype=tf.int32) * bos_id
    prefix_len = tf.ones([batch_size], dtype=tf.int32)
  else:
    assert int(prefix.shape[0]) == batch_size, (batch_size, prefix.shape)
    assert int(prefix_len.shape[0]) == batch_size, (batch_size,
                                                    prefix_len.shape)
    output_len += int(prefix.shape[1])

  if debug:
    tpu_summary.tensor('prefix', prefix)
    tpu_summary.tensor('prefix_len', prefix_len)

  with tf.name_scope('init_state'):
    t = tf.constant(0)
    tgt_id = tf.zeros([batch_size, beam_size], dtype=tf.int32)
    tgt_id += bos_id
    tgt_pos = tf.zeros([batch_size, beam_size], dtype=tf.int32)
    tgt_mask = tf.zeros([batch_size, beam_size, buf_size], dtype=fprop_dtype)
    tgt_mask += tf.one_hot(tf.range(beam_size), buf_size, dtype=fprop_dtype)
    hyp_score = tf.zeros([batch_size, beam_size], dtype=fprop_dtype)
    # penalize all hyps except the first
    hyp_score -= tf.cast(
        tf.range(beam_size, dtype=tf.float32) * 1e5, dtype=fprop_dtype)
    nbest_size = nbest_size or beam_size
    nbest_score = tf.zeros([batch_size, nbest_size], dtype=fprop_dtype)
    nbest_score -= 1e9
    nbest_score_norm = nbest_score
    nbest_mask = tf.zeros([batch_size, nbest_size, buf_size], dtype=fprop_dtype)

  with tf.name_scope('init_ext'):
    # Initialize the extension buffer.
    #
    # Extension buffer stores a (potentially large) set of 'extensions',
    # which consist of a hypothesis (represented by ext_mask) and next token
    # (represented by ext_id). At each decoder iteration, top_k extensions
    # from each hypothesis are added to the buffer and sorted by score.
    #
    # Then top beam_size extensions are removed from the buffer and used
    # in the next decoder iteration. And top 'ext_size' remaining extensions
    # are carried over to be possibly evaluated at a later step.
    #
    # As a result of this manipulation, the decoder is no longer restricted
    # to always compare hyps of the same token length at each iteration.
    # In particular, for a fixed length N it can generate more than beam_size
    # terminated hyps.
    #
    # Setting ext_size = 0 disables this feautre.
    if ext_size:
      ext_id = tf.zeros([batch_size, ext_size], dtype=tf.int32)
      ext_score = tf.zeros([batch_size, ext_size], dtype=fprop_dtype)
      ext_score -= 1e9
      ext_mask = tf.zeros([batch_size, ext_size, buf_size], dtype=fprop_dtype)
    else:
      ext_size = ext_id = ext_score = ext_mask = 0

  with tf.name_scope('init_prefix'):
    # rename prefix->pfx for shorter variables
    pfx = tf.cast(prefix, tf.int32)
    pfx_len = tf.cast(prefix_len, tf.int32)
    del prefix, prefix_len
    # Before the first call to dec_callback() the prefix shall be packed into
    # the tgt_id buffer as follows:
    #
    # [ - - - - - - P P P P P P P* - - - ]   ^
    # [ - - P P P P P P P P P P P* - - - ]   | batch
    # [ - - - - - - - - - - - P P* - - - ]   V
    # |<---- prefix len ---->  |<-- beam -->
    #
    # The last meaningful token in the prefix (P*)
    # must be located at the same position in all batch rows.
    #
    # We then make one dec_callback() with full prefix (minus P*)
    # which will populate the initial dec_state
    # (for transformer -- self-attention key/value cache)
    #
    # The last block [batch, beam] then becomes the first tgt_id for the loop.
    pfx_max = int(pfx.shape[1])
    pfx_mul = pfx_max // beam_size
    assert pfx_max == pfx_mul * beam_size, (pfx_max, pfx_mul, beam_size)
    pfx_time = tf.range(pfx_max)
    pfx_indexes = pfx_time - pfx_max + tf.expand_dims(pfx_len - 1, 1)
    pfx_pad = tf.cast(tf.greater_equal(pfx_indexes, 0),
                      tf.int32)  # Exclude final pfx token.
    pfx_id = tf.roll(pfx, shift=1, axis=-1) * pfx_pad
    pfx_last = pfx[:, -1]

    buf_time = tf.range(buf_size)
    pfx_time_mask = tf.cast(
        tf.less_equal(tf.expand_dims(buf_time, 0), tf.expand_dims(pfx_time, 1)),
        fprop_dtype)
    pfx_mask = tf.einsum('BQ,QK->BQK', tf.cast(pfx_pad, fprop_dtype),
                         pfx_time_mask)
    # Remove padding.
    assert buf_size > pfx_max
    pfx_pad_long = tf.pad(
        pfx_pad, [(0, 0), (0, buf_size - pfx_max)], constant_values=1)
    pfx_mask *= tf.cast(tf.expand_dims(pfx_pad_long, axis=1), fprop_dtype)
    pfx_segment_id = pfx_pad
    pfx_pos = pfx_indexes * pfx_pad

    if debug:
      tpu_summary.tensor('pfx_id', pfx_id)
      tpu_summary.tensor('pfx_len', pfx_len)
      tpu_summary.tensor('pfx_pos', pfx_pos)
      tpu_summary.tensor('pfx_last', pfx_last)

    # Now call decoder with prefix minus P*:
    # 'dec_state' now shall contain the key/value cache for prefix tokens
    # (for transformer models), and 'logits' we can either discard or
    # roll into the initial hyp_score. Discard is simpler.
    with tf.name_scope('prefix_fprop'):
      # TODO(krikun): remove extra type checks
      assert (pfx_id.dtype == tf.int32), (pfx_id.dtype)
      assert (pfx_segment_id.dtype == tf.int32), (pfx_segment_id.dtype)
      assert (pfx_pos.dtype == tf.int32), (pfx_pos.dtype)
      assert (pfx_mask.dtype == fprop_dtype), (pfx_mask.dtype)
      assert (t.dtype == tf.int32), (t.dtype)
      logits, dec_state = dec_callback(pfx_id, pfx_segment_id, pfx_pos,
                                       pfx_mask, dec_state, t)
      del logits

    # Now construct the initial state for the rest of the beam search loop.
    # 'tgt_id' is simply 'pfx_last' padded to [batch, beam] shape
    # 'tgt_pos' is different for each batch row and is equal to prefix_len
    # 'tgt_segment_id' always 1 (no packing)
    # 'hyp_score' is 0 for beam=0 and negative for beam>=1
    tgt_id = tf.zeros([batch_size, beam_size], tf.int32) + tf.expand_dims(
        pfx_last, 1)
    tgt_pos = tf.zeros([batch_size, beam_size], tf.int32) + tf.expand_dims(
        (pfx_len - 1), 1)
    hyp_score = tf.zeros([batch_size, beam_size], dtype=fprop_dtype) - tf.cast(
        tf.range(beam_size, dtype=tf.float32) * 1e5, dtype=fprop_dtype)

    # TODO(krikun) Here we make initial 't' constant and determined by the
    # shape of the prefix tensor 'pfx_max'. It is possible to make it dynamic
    # as t ~  max(pfx_len) / beam_size and this will more steps for beam search
    # however 'max' results in a very slow all-to-all for 'max' on 16x16
    # and variable number of decoder steps may result in bad latency.
    t = tf.cast(tf.math.ceil(pfx_max / beam_size), tf.int32)

    # Initial tgt_mask is such that each token P* has attention on itself
    # (as usual) and on all prefix tokens before it, which are not padding.
    tgt_mask = tf.zeros([batch_size, beam_size, buf_size], dtype=fprop_dtype)
    tgt_mask += tf.cast(
        tf.expand_dims(tf.pad(pfx_pad, [[0, 0], [0, (buf_size - pfx_max)]]), 1),
        fprop_dtype)
    tgt_mask += tf.one_hot(
        tf.range(beam_size) + t * beam_size, buf_size, dtype=fprop_dtype)

    if debug:
      tpu_summary.tensor('tgt_id', tgt_id)
      tpu_summary.tensor('tgt_pos', tgt_pos)
      tpu_summary.tensor('tgt_mask', tgt_mask)
      tpu_summary.tensor('t', t)

  with tf.name_scope('init_hist'):
    # h_tgt_id is used to recover topk_ids from nbest_mask
    h_tgt_id = tf.TensorArray(dtype=tf.int32, size=max_steps)
    h_tgt_pos = tf.TensorArray(dtype=tf.int32, size=max_steps)

    # When non-trivial prefix is present we also write prefix ids to
    # h_tgt_id so that the full sequence including prefix can be recovered
    # by unmask() below.  When prefix is empty, pfx_id shape is [batch, 0]
    # and the loop below becomes a no-op.
    # TODO(krikun): maybe a tf.while_loop is more appropriate here.
    for i, x_i in enumerate(tf.split(pfx_id, pfx_mul, 1)):
      h_tgt_id = h_tgt_id.write(i, x_i)
    for i, x_i in enumerate(tf.split(pfx_pos, pfx_mul, 1)):
      h_tgt_pos = h_tgt_pos.write(i, x_i)

    hist = (h_tgt_id, h_tgt_pos)
    tf.logging.info('hist=%r', hist)

  nbest_hyps = (nbest_mask, nbest_score, nbest_score_norm)
  tf.logging.info('nbest_hyps=%r', nbest_hyps)

  ext = (ext_id, ext_score, ext_mask)
  tf.logging.info('ext=%r', ext)

  loop_vars = (t, tgt_id, tgt_pos, tgt_mask, hyp_score, nbest_hyps, ext, hist)
  tf.logging.info('loop_vars=%r', loop_vars)

  def loop_step(loop_vars, dec_state):  # pylint: disable=missing-docstring
    tf.logging.info('loop_vars=%r', loop_vars)
    tf.logging.info('dec_state=%r', dec_state)
    (t, tgt_id, tgt_pos, tgt_mask, hyp_score, nbest_hyps, ext, hist) = loop_vars
    (ext_id, ext_score, ext_mask) = ext
    (h_tgt_id, h_tgt_pos) = hist
    h_tgt_id = h_tgt_id.write(t, tgt_id, name='h_tgt_id')
    h_tgt_pos = h_tgt_pos.write(t, tgt_pos, name='h_tgt_pos')
    # not using tf.ones() here because of XLA compilation error
    tgt_segment_id = tgt_id * 0 + 1
    logits, dec_state = dec_callback(tgt_id, tgt_segment_id, tgt_pos, tgt_mask,
                                     dec_state, t)
    # take predicted EOS score for each hyp and compute normalized score
    eos_score = hyp_score + tf.cast(logits[:, :, eos_id], hyp_score.dtype)

    def length_norm(t):
      t = tf.cast(t, fprop_dtype)
      alpha = length_norm_alpha
      tf.logging.info('length_norm.alpha=%r', alpha)
      return tf.math.pow((t + 5.) / 5., alpha)

    hyp_len = tgt_pos - tf.expand_dims((pfx_len - 1), -1)
    eos_score_norm = eos_score / length_norm(hyp_len)
    # update the n-best list
    nbest_hyps = update_nbest(nbest_hyps, (tgt_mask, hyp_score, eos_score_norm))

    if debug:
      tpu_summary.tensor('eos_score', eos_score)
      tpu_summary.tensor('hyp_len', hyp_len)

    # take top k tokens for each hyp
    k = beam_size
    with tf.name_scope('topk1'):
      top_score, top_id = top_k_fn(logits, k)
      top_score = tf.cast(top_score, fprop_dtype)

    top_score += tf.expand_dims(hyp_score, -1)
    top_score -= 1e9 * tf.cast(tf.equal(top_id, eos_id), fprop_dtype)

    top_score = tf.reshape(top_score, [batch_size, beam_size * k])
    top_id = tf.reshape(top_id, [batch_size, beam_size * k])
    top_mask = tf.repeat(tgt_mask, beam_size, 1)

    if debug:
      tpu_summary.tensor('top_id', top_id)
      tpu_summary.tensor('top_score', top_score)
      # tpu_summary.tensor('top_mask', top_mask)

    with tf.name_scope('update_ext'):
      # combine top k tokens with extension buffer (if any)
      if ext_size:
        ext_id = tf.concat([ext_id, top_id], 1)
        ext_score = tf.concat([ext_score, top_score], 1)
        ext_mask = tf.concat([ext_mask, top_mask], 1)
      else:
        ext_id, ext_score, ext_mask = top_id, top_score, top_mask

      # sort by score
      ext_score, i = tf.math.top_k(ext_score, ext_size + beam_size)
      i1 = tf.one_hot(i, ext_size + beam_size * k, dtype=fprop_dtype)
      ext_mask = tf.einsum('bkt,bjk->bjt', ext_mask, i1)
      ext_id = einsum_i32('bk,bjk->bj', ext_id, i1)

      # pick top beam_size extensions to evaluate at next iteration
      if ext_size:
        hyp_score = ext_score[:, :beam_size]
        ext_score = ext_score[:, beam_size:]
        tgt_id = ext_id[:, :beam_size]
        ext_id = ext_id[:, beam_size:]
        tgt_mask = ext_mask[:, :beam_size]
        ext_mask = ext_mask[:, beam_size:]
      else:
        hyp_score, tgt_id, tgt_mask = ext_score, ext_id, ext_mask
        ext_score = ext_id = ext_mask = 0

    tgt_pos = tf.reduce_sum(tgt_mask, -1)
    tgt_pos = tf.cast(tgt_pos, tf.int32)

    t += 1
    with tf.name_scope('tgt_mask_extend'):
      tgt_mask += tf.one_hot(
          tf.range(beam_size) + t * beam_size, buf_size, dtype=fprop_dtype)

    ext = (ext_id, ext_score, ext_mask)
    hist = (h_tgt_id, h_tgt_pos)
    loop_vars = (t, tgt_id, tgt_pos, tgt_mask, hyp_score, nbest_hyps, ext, hist)
    tf.logging.info('loop_vars=%r', loop_vars)
    tf.logging.info('dec_state=%r', dec_state)
    return loop_vars, dec_state

  def loop_cond(loop_vars, dec_state):  # pylint: disable=missing-docstring
    tf.logging.info('loop_vars=%r', loop_vars)
    tf.logging.info('dec_state=%r', dec_state)
    if beam_gap is None:
      (t, _, _, _, _, _, _, _) = loop_vars
      return t < max_steps
    else:
      (t, _, _, _, _, nbest_hyps, _, _) = loop_vars
      (_, nbest_score, _) = nbest_hyps
      # stop early if all current hyps are significantly worse than nbest
      diff = tf.reduce_min(
          tf.reduce_min(nbest_score, -1) - tf.reduce_max(hyp_score, -1))
      return tf.math.logical_and(t < max_steps, diff < beam_gap)

  with tf.name_scope('flat_beam_search_loop'):
    (loop_vars, dec_state) = tf.while_loop(
        loop_cond,
        loop_step,
        loop_vars=(loop_vars, dec_state),
        back_prop=False,
        swap_memory=False,
        maximum_iterations=max_steps)

  # flatten all tensorarrays into tensors
  (t, tgt_id, tgt_pos, tgt_mask, hyp_score, nbest_hyps, ext, hist) = loop_vars
  (nbest_mask, nbest_score, nbest_score_norm) = nbest_hyps
  (h_tgt_id, h_tgt_pos) = hist
  h_tgt_id = h_tgt_id.stack()
  h_tgt_pos = h_tgt_pos.stack()
  hist = (h_tgt_id, h_tgt_pos)
  loop_vars = (t, tgt_id, tgt_pos, tgt_mask, hyp_score, nbest_hyps, ext, hist)

  # recover topk_ids from nbest_mask and tgt_id history
  h = tf.transpose(h_tgt_id, [1, 0, 2])
  h = tf.reshape(h, [batch_size, buf_size])

  def unmask(h, m):
    with tf.name_scope('unmask'):
      tpu_summary.tensor('unmask_h', h)
      tpu_summary.tensor('unmask_m', m)
      t = tf.cumsum(m, -1) * m - 1
      mh = einsum_i32('bkt,bt->bkt', m, h)
      t2 = tf.one_hot(tf.cast(t, tf.int32), output_len, dtype=fprop_dtype)
      x = einsum_i32('bkt,bktT->bkT', mh, t2)
      return tf.cast(x, h.dtype)

  topk_ids = unmask(h, nbest_mask)
  topk_len = tf.reduce_sum(nbest_mask, -1)
  topk_len = tf.cast(topk_len, tf.int32)
  # add eos, because nbest_mask does not encode eos
  topk_ids += eos_id * tf.one_hot(topk_len, output_len, dtype=tf.int32)
  topk_len += 1
  topk_len = tf.minimum(topk_len, output_len)
  topk_score = nbest_score_norm

  nbest = (topk_ids, topk_len, topk_score)

  return loop_vars, dec_state, nbest
