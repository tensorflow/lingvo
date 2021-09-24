# Lint as: python3
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
"""Tests for flat_beam_search_helper."""

from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import flat_beam_search_helper
from lingvo.core import test_utils
from lingvo.core import tpu_summary
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
# pylint: enable=g-direct-tensorflow-import


class TestDecoder:
  """Decoder with state which produces well-defined logits.

  It determines the best next token according to one of the rules:
  - increment last token id by 1 (rule='+1')
  - take sum of all previous tokens (rule='sum')
  - take sum of last two tokens (rule='fib')

  With small probability it will make an error and an extra +1.
  EOS is predicted with high probability after any token id that ends with 9.
  """

  def __init__(self, batch_size, beam_size, max_steps, vocab_size, rule):
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.beam_size = beam_size
    self.max_steps = max_steps
    self.rule = rule
    assert rule in ('+1', 'sum', 'fib')

  def new_state(self):
    buf = tf.zeros([self.max_steps, self.batch_size, self.beam_size],
                   tf.int32,
                   name='buf')
    dec_state = [buf]
    return dec_state

  def dec_callback(self, tgt_id, tgt_pos, tgt_segment_id, tgt_mask, dec_state,
                   t):
    del tgt_pos, tgt_segment_id

    [buf] = dec_state
    if tgt_id.shape == (self.batch_size, self.beam_size):
      buf = inplace_ops.alias_inplace_update(buf, t, tgt_id)
    else:
      div = int(tgt_id.shape[1] // self.beam_size)
      for i, x_i in enumerate(tf.split(tgt_id, div, 1)):
        buf = inplace_ops.alias_inplace_update(buf, t + i, x_i)

    buf1 = tf.transpose(buf, [1, 0, 2])
    buf1 = tf.reshape(buf1, [self.batch_size, self.max_steps * self.beam_size])

    # select next_tgt_id as a function of previous target tokens
    if self.rule == '+1':
      next_tgt_id = (tgt_id + 1)
      next_tgt_id %= self.vocab_size
    elif self.rule == 'sum':
      # sum over all previous tokens in tgt_mask
      next_tgt_id = tf.einsum('BT,BKT->BK', buf1, tf.cast(tgt_mask, tf.int32))
      next_tgt_id %= self.vocab_size
    elif self.rule == 'fib':
      # select last token according to tgt_mask
      m = tgt_mask
      m *= tf.cast(
          tf.equal(tf.cumsum(m, -1),
                   tf.reduce_sum(m, -1, keepdims=True) - 1), m.dtype)
      last_tgt_id = tf.einsum('BT,BKT->BK', buf1, tf.cast(m, tf.int32))
      next_tgt_id = (last_tgt_id + tgt_id) % self.vocab_size

    # with a lower probably add extra +1 to the correct next_tgt_id
    n = self.vocab_size
    logits = 5 * tf.one_hot(next_tgt_id % n, n)
    logits += 4 * tf.one_hot((next_tgt_id + 1) % n, n)
    logits += 3 * tf.one_hot((next_tgt_id + 2) % n, n)
    logits += 2 * tf.one_hot((next_tgt_id + 3) % n, n)
    logits += 1 * tf.one_hot((next_tgt_id + 4) % n, n)

    # increase eos_score if current tgt_id contains 9
    eos_id = 0
    tgt_id_contains_9 = tf.logical_or(
        tf.equal(tgt_id % 10, 9), tf.equal((tgt_id // 10) % 10, 9))
    logits += 9 * tf.einsum('V,BK->BKV', tf.one_hot(eos_id, self.vocab_size),
                            tf.cast(tgt_id_contains_9, tf.float32))

    # tie-breaking -- lower token id wins a little bit
    tie = np.arange(0., 1., 1. / n)
    tie /= tie.sum()
    logits -= tie

    logits = tf.nn.log_softmax(logits)

    dec_state = [buf]
    return logits, dec_state


class FlatBeamSearchTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters({'rule': '+1'}, {'rule': 'sum'}, {'rule': 'fib'})
  def testFlatBeamSearch(self, rule):
    batch_size = 2
    beam_size = 4
    max_steps = 20
    vocab_size = 100

    decoder = TestDecoder(batch_size, beam_size, max_steps, vocab_size, rule)
    dec_state = decoder.new_state()
    dec_callback = decoder.dec_callback

    with tpu_summary.context(rewrite_while_loop=True):
      bs = flat_beam_search_helper.flat_beam_search(
          batch_size,
          beam_size,
          max_steps,
          dec_callback,
          dec_state,
          bos_id=1,
          eos_id=0,
          beam_gap=None,
          debug=True)
      debug_tensors = tpu_summary.merge_all()

    tf.logging.info('bs=%r', bs)
    tf.logging.info('debug_tensors=%r', debug_tensors)

    with self.session() as sess:
      [bs, debug_tensors] = sess.run([bs, debug_tensors])

    tf.logging.info('bs=%r', bs)

    loop_vars, dec_state_, nbest = bs
    (topk_ids, topk_lens, topk_scores) = nbest
    del loop_vars, dec_state_, nbest

    self.assertEqual((batch_size, beam_size, max_steps), topk_ids.shape)
    self.assertEqual((batch_size, beam_size), topk_lens.shape)
    self.assertEqual((batch_size, beam_size), topk_scores.shape)

    print('Decoder output rule=%r' % decoder.rule)
    print('batch_size=%d beam_size=%d max_steps=%d' %
          (batch_size, beam_size, max_steps))

    topk = [[
        topk_ids[b, k, 0:topk_lens[b, k]].tolist() for k in range(beam_size)
    ] for b in range(batch_size)]

    for b in range(batch_size):
      for k in range(beam_size):
        print('topk[%d][%d] (%0.6f): %r' %
              (b, k, topk_scores[b, k], topk[b][k]))

    # pyformat: disable
    if decoder.rule == '+1':
      expected = 2 * [[
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
          [1, 2, 3, 4, 5, 6, 7, 9, 0],
          [1, 2, 3, 4, 5, 6, 8, 9, 0],
          [1, 2, 3, 4, 5, 7, 8, 9, 0],
      ]]
    elif decoder.rule == 'sum':
      expected = 2 * [[
          [1, 1, 2, 4, 9, 0],
          [1, 1, 2, 5, 9, 0],
          [1, 1, 2, 4, 8, 16, 32, 64, 29, 0],
          [1, 1, 2, 4, 8, 16, 32, 65, 29, 0],
      ]]
    elif decoder.rule == 'fib':
      expected = 2 * [[
          [1, 1, 2, 3, 5, 9, 0],
          [1, 1, 2, 3, 6, 9, 0],
          [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 0],
          [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 90, 0],
      ]]
    # pyformat: enable

    self.assertEqual(expected, topk)

  @parameterized.parameters({'rule': '+1'}, {'rule': 'sum'}, {'rule': 'fib'})
  def testFlatBeamSearchWithPrefix(self, rule):
    batch_size = 2
    beam_size = 4
    max_steps = 20
    vocab_size = 100
    prefix_size = 4

    prefix_len = np.zeros([batch_size])
    prefix_len[0] = 2
    prefix_len[1] = 3
    prefix_id = np.zeros([batch_size, prefix_size])
    prefix_id[0, :2] = [11, 12]
    prefix_id[1, :3] = [21, 22, 23]

    decoder = TestDecoder(batch_size, beam_size, max_steps, vocab_size, rule)
    dec_state = decoder.new_state()
    dec_callback = decoder.dec_callback

    with tpu_summary.context(rewrite_while_loop=True):
      bs = flat_beam_search_helper.flat_beam_search(
          batch_size,
          beam_size,
          max_steps,
          dec_callback,
          dec_state,
          bos_id=1,
          eos_id=0,
          prefix=prefix_id,
          prefix_len=prefix_len,
          beam_gap=None,
          debug=True)
      debug_tensors = tpu_summary.merge_all()

    tf.logging.info('bs=%r', bs)
    tf.logging.info('debug_tensors=%r', debug_tensors)

    with self.session() as sess:
      [bs, debug_tensors] = sess.run([bs, debug_tensors])

    tf.logging.info('bs=%r', bs)

    loop_vars, dec_state_, nbest = bs
    (topk_ids, topk_lens, topk_scores) = nbest
    del loop_vars, dec_state_

    self.assertEqual((batch_size, beam_size, max_steps + prefix_size),
                     topk_ids.shape)
    self.assertEqual((batch_size, beam_size), topk_lens.shape)
    self.assertEqual((batch_size, beam_size), topk_scores.shape)

    print('Decoder output rule=%r' % decoder.rule)
    print('batch_size=%d beam_size=%d max_steps=%d' %
          (batch_size, beam_size, max_steps))

    topk = [[
        topk_ids[b, k, 0:topk_lens[b, k]].tolist() for k in range(beam_size)
    ] for b in range(batch_size)]

    for b in range(batch_size):
      for k in range(beam_size):
        print('topk[%d][%d] (%0.6f): %r' %
              (b, k, topk_scores[b, k], topk[b][k]))

    # pyformat: disable
    if decoder.rule == '+1':
      expected = [
          [[11, 12, 13, 14, 15, 16, 17, 18, 19, 0],
           [11, 12, 13, 14, 15, 16, 17, 19, 0],
           [11, 12, 13, 14, 15, 16, 18, 19, 0],
           [11, 12, 13, 14, 15, 17, 18, 19, 0]],
          [[21, 22, 23, 24, 25, 26, 27, 28, 29, 0],
           [21, 22, 23, 24, 25, 26, 27, 29, 0],
           [21, 22, 23, 24, 25, 26, 28, 29, 0],
           [21, 22, 23, 24, 25, 27, 28, 29, 0]]]
    elif decoder.rule == 'sum':
      expected = [
          [[11, 12, 23, 46, 92, 0],
           [11, 12, 23, 46, 93, 0],
           [11, 12, 23, 47, 93, 0],
           [11, 12, 24, 47, 94, 0]],
          [[21, 22, 23, 66, 32, 64, 29, 0],
           [21, 22, 23, 66, 32, 65, 29, 0],
           [21, 22, 23, 66, 32, 64, 28, 56, 12, 24, 48, 96, 0],
           [21, 22, 23, 69, 0]]]
    elif decoder.rule == 'fib':
      expected = [
          [[11, 12, 23, 35, 58, 93, 0],
           [11, 12, 23, 35, 59, 0],
           [11, 12, 23, 36, 59, 0],
           [11, 12, 23, 35, 58, 94, 0]],
          [[21, 22, 23, 45, 69, 0],
           [21, 22, 23, 46, 69, 0],
           [21, 22, 23, 45, 68, 13, 81, 94, 0],
           [21, 22, 23, 45, 68, 13, 81, 95, 0]]]
    # pyformat: enable

    # locals().update({k.split('/')[0]:v for k,v in debug_tensors.items()})
    # import ipdb; ipdb.set_trace()

    self.assertEqual(expected, topk)

  @parameterized.parameters({'rule': '+1'}, {'rule': 'sum'}, {'rule': 'fib'})
  def testFlatBeamSearchWithExtensionBuffer(self, rule):
    batch_size = 2
    beam_size = 4
    ext_size = 128
    nbest_size = 8
    max_steps = 300
    vocab_size = 100

    decoder = TestDecoder(batch_size, beam_size, max_steps, vocab_size, rule)
    dec_state = decoder.new_state()
    dec_callback = decoder.dec_callback

    with tpu_summary.context(rewrite_while_loop=True):
      bs = flat_beam_search_helper.flat_beam_search(
          batch_size,
          beam_size,
          max_steps,
          dec_callback,
          dec_state,
          bos_id=1,
          eos_id=0,
          beam_gap=None,
          ext_size=ext_size,
          nbest_size=nbest_size,
          debug=True)
      debug_tensors = tpu_summary.merge_all()

    tf.logging.info('bs=%r', bs)
    tf.logging.info('debug_tensors=%r', debug_tensors)

    with self.session() as sess:
      [bs, debug_tensors] = sess.run([bs, debug_tensors])

    tf.logging.info('bs=%r', bs)

    loop_vars, dec_state_, nbest = bs
    (topk_ids, topk_lens, topk_scores) = nbest
    del loop_vars, dec_state_, nbest

    self.assertEqual((batch_size, nbest_size, max_steps), topk_ids.shape)
    self.assertEqual((batch_size, nbest_size), topk_lens.shape)
    self.assertEqual((batch_size, nbest_size), topk_scores.shape)

    print('Decoder output rule=%r' % decoder.rule)
    print('batch_size=%d beam_size=%d ext_size=%d nbest_size=%d max_steps=%d' %
          (batch_size, beam_size, ext_size, nbest_size, max_steps))

    topk = [[
        topk_ids[b, k, 0:topk_lens[b, k]].tolist() for k in range(nbest_size)
    ] for b in range(batch_size)]

    for b in range(batch_size):
      for k in range(nbest_size):
        print('topk[%d][%d] (%0.6f): %r' %
              (b, k, topk_scores[b, k], topk[b][k]))

    # pyformat: disable
    if decoder.rule == '+1':
      expected = 2 * [[
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
          [1, 2, 3, 4, 5, 6, 7, 9, 0],
          [1, 2, 3, 4, 5, 6, 8, 9, 0],
          [1, 2, 3, 4, 5, 7, 8, 9, 0],
          [1, 2, 3, 4, 6, 7, 8, 9, 0],
          [1, 2, 3, 5, 6, 7, 8, 9, 0],
          [1, 2, 4, 5, 6, 7, 8, 9, 0],
          [1, 3, 4, 5, 6, 7, 8, 9, 0],
      ]]
    elif decoder.rule == 'sum':
      expected = 2 * [[
          [1, 1, 2, 4, 9, 0],
          [1, 1, 2, 5, 9, 0],
          [1, 2, 3, 6, 12, 24, 48, 96, 0],
          [1, 1, 2, 4, 8, 16, 32, 64, 29, 0],
          [1, 1, 2, 4, 8, 16, 32, 65, 29, 0],
          [1, 1, 2, 5, 10, 19, 0],
          [1, 2, 3, 6, 12, 25, 49, 0],
          [1, 2, 3, 6, 12, 24, 49, 0],
      ]]
    elif decoder.rule == 'fib':
      expected = 2 * [[
          [1, 1, 2, 3, 5, 9, 0],
          [1, 1, 2, 3, 6, 9, 0],
          [1, 2, 3, 5, 9, 0],
          [1, 1, 4, 5, 9, 0],
          [1, 2, 3, 6, 9, 0],
          [1, 1, 3, 4, 7, 11, 18, 29, 0],
          [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 0],
          [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 0],
      ]]
    # pyformat: enable

    self.assertEqual(expected, topk)


if __name__ == '__main__':
  tf.test.main()
