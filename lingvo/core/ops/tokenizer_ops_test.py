# -*- coding: utf-8 -*-
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
# ==============================================================================
"""Tests for tokenizer_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.core.ops import py_x_ops


class TokenizerOpsTest(test_utils.TestCase):

  def testLabelsToTokenId(self):
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.ascii_to_token_id([
              'hElLo', 'sIr<epsilon>', 'What a <unk> day', 'america\'s',
              '<noise> early', '1:00 AM', '<text_only>morning'
          ],
                                     append_eos=True,
                                     maxlen=10))
    self.assertAllEqual(token_ids, [
        [1, 12, 9, 16, 16, 19, 2, 2, 2, 2],
        [1, 23, 13, 22, 73, 2, 2, 2, 2, 2],
        [1, 27, 12, 5, 24, 3, 5, 3, 0, 3],
        [1, 5, 17, 9, 22, 13, 7, 5, 32, 23],
        [1, 4, 3, 9, 5, 22, 16, 29, 2, 2],
        [1, 40, 34, 39, 39, 3, 5, 17, 2, 2],
        [1, 74, 17, 19, 22, 18, 13, 18, 11, 2],
    ])
    self.assertAllEqual(
        target_ids,
        [[12, 9, 16, 16, 19, 2, 2, 2, 2, 2], [23, 13, 22, 73, 2, 2, 2, 2, 2, 2],
         [27, 12, 5, 24, 3, 5, 3, 0, 3, 2], [5, 17, 9, 22, 13, 7, 5, 32, 23, 2],
         [4, 3, 9, 5, 22, 16, 29, 2, 2, 2], [40, 34, 39, 39, 3, 5, 17, 2, 2, 2],
         [74, 17, 19, 22, 18, 13, 18, 11, 2, 2]])
    self.assertAllEqual(
        paddings,
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

  def testLabelsToTokenIdAppendEOSFalse(self):
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.ascii_to_token_id([
              'hElLo', 'sIr<epsilon>', 'What a <unk> day', 'america\'s',
              '<noise> early', '1:00 AM', '100%'
          ],
                                     append_eos=False,
                                     maxlen=10))
    self.assertAllEqual(
        token_ids,
        [[1, 12, 9, 16, 16, 19, 2, 2, 2, 2], [1, 23, 13, 22, 73, 2, 2, 2, 2, 2],
         [1, 27, 12, 5, 24, 3, 5, 3, 0, 3], [1, 5, 17, 9, 22, 13, 7, 5, 32, 23],
         [1, 4, 3, 9, 5, 22, 16, 29, 2, 2], [1, 40, 34, 39, 39, 3, 5, 17, 2, 2],
         [1, 40, 39, 39, 52, 2, 2, 2, 2, 2]])
    self.assertAllEqual(
        target_ids,
        [[12, 9, 16, 16, 19, 2, 2, 2, 2, 2], [23, 13, 22, 73, 2, 2, 2, 2, 2, 2],
         [27, 12, 5, 24, 3, 5, 3, 0, 3, 2], [5, 17, 9, 22, 13, 7, 5, 32, 23, 2],
         [4, 3, 9, 5, 22, 16, 29, 2, 2, 2], [40, 34, 39, 39, 3, 5, 17, 2, 2, 2],
         [40, 39, 39, 52, 2, 2, 2, 2, 2, 2]])
    self.assertAllEqual(
        paddings,
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        ], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

  def testLabelsToTokenIdNoPadToMaxlen(self):
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.ascii_to_token_id([
              'hElLo', 'sIr<epsilon>', 'What a <unk> day', 'america\'s',
              '<noise> early', '1:00 AM', '<text_only>morning'
          ],
                                     append_eos=True,
                                     maxlen=20,
                                     pad_to_maxlen=False))
    self.assertAllEqual(token_ids, [
        [1, 12, 9, 16, 16, 19, 2, 2, 2, 2, 2, 2, 2],
        [1, 23, 13, 22, 73, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 27, 12, 5, 24, 3, 5, 3, 0, 3, 8, 5, 29],
        [1, 5, 17, 9, 22, 13, 7, 5, 32, 23, 2, 2, 2],
        [1, 4, 3, 9, 5, 22, 16, 29, 2, 2, 2, 2, 2],
        [1, 40, 34, 39, 39, 3, 5, 17, 2, 2, 2, 2, 2],
        [1, 74, 17, 19, 22, 18, 13, 18, 11, 2, 2, 2, 2],
    ])
    self.assertAllEqual(target_ids, [
        [12, 9, 16, 16, 19, 2, 2, 2, 2, 2, 2, 2, 2],
        [23, 13, 22, 73, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [27, 12, 5, 24, 3, 5, 3, 0, 3, 8, 5, 29, 2],
        [5, 17, 9, 22, 13, 7, 5, 32, 23, 2, 2, 2, 2],
        [4, 3, 9, 5, 22, 16, 29, 2, 2, 2, 2, 2, 2],
        [40, 34, 39, 39, 3, 5, 17, 2, 2, 2, 2, 2, 2],
        [74, 17, 19, 22, 18, 13, 18, 11, 2, 2, 2, 2, 2],
    ])
    self.assertAllEqual(paddings, [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])

  def testIdToToken(self):
    with self.session(use_gpu=False) as sess:
      token_ids = [[12, 9, 16, 16, 19, 2, 2, 2, 2,
                    2], [23, 13, 22, 73, 2, 2, 2, 2, 2,
                         2], [27, 12, 5, 24, 3, 5, 3, 0, 3,
                              2], [5, 17, 9, 22, 13, 7, 5, 32, 23, 2],
                   [4, 3, 9, 5, 22, 16, 29, 2, 2,
                    2], [40, 34, 39, 39, 3, 5, 17, 2, 2,
                         2], [52, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
      seq_lens = [5, 4, 9, 9, 7, 7, 1]
      tokens = sess.run(py_x_ops.id_to_ascii(token_ids, seq_lens))

    self.assertEqual(tokens.tolist(), [
        'hello', 'sir<epsilon>', 'what a <unk> ', "america's", '<noise> early',
        '1:00 am', '%'
    ])

  def testStrToVocabToken(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.str_to_vocab_tokens(
              [
                  'a b c d e',
                  '<epsilon> <S> </S> <UNK>',
                  'øut über ♣ 愤青 ←',
              ],
              append_eos=True,
              maxlen=10,
              vocab_filepath=vocab))
      self.assertEqual(
          token_ids.tolist(),
          [[1, 5, 6, 7, 8, 9, 2, 2, 2, 2], [1, 0, 1, 2, 3, 2, 2, 2, 2, 2],
           [1, 10, 11, 12, 13, 3, 2, 2, 2, 2]])
      self.assertEqual(
          target_ids.tolist(),
          [[5, 6, 7, 8, 9, 2, 2, 2, 2, 2], [0, 1, 2, 3, 2, 2, 2, 2, 2, 2],
           [10, 11, 12, 13, 3, 2, 2, 2, 2, 2]])
      self.assertEqual(paddings.tolist(),
                       [[0., 0., 0., 0., 0., 0., 1., 1., 1., 1.], [
                           0., 0., 0., 0., 0., 1., 1., 1., 1., 1.
                       ], [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]])

  def testStrToVocabTokenAppendEOSFalse(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.str_to_vocab_tokens(
              [
                  'a b c d e',
                  '<epsilon> <S> </S> <UNK>',
                  'øut über ♣ 愤青 ←',
              ],
              append_eos=False,
              maxlen=10,
              vocab_filepath=vocab))
      self.assertEqual(
          token_ids.tolist(),
          [[1, 5, 6, 7, 8, 9, 2, 2, 2, 2], [1, 0, 1, 2, 3, 2, 2, 2, 2, 2],
           [1, 10, 11, 12, 13, 3, 2, 2, 2, 2]])
      self.assertEqual(
          target_ids.tolist(),
          [[5, 6, 7, 8, 9, 2, 2, 2, 2, 2], [0, 1, 2, 3, 2, 2, 2, 2, 2, 2],
           [10, 11, 12, 13, 3, 2, 2, 2, 2, 2]])
      self.assertEqual(paddings.tolist(),
                       [[0., 0., 0., 0., 0., 1., 1., 1., 1., 1.], [
                           0., 0., 0., 0., 1., 1., 1., 1., 1., 1.
                       ], [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]])

  def testStrToVocabTokenTruncates(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.str_to_vocab_tokens(['a b c d e ' * 1000],
                                       append_eos=True,
                                       maxlen=5,
                                       vocab_filepath=vocab))
      self.assertEqual(token_ids.tolist(), [[1, 5, 6, 7, 8]])
      self.assertEqual(target_ids.tolist(), [[5, 6, 7, 8, 9]])
      self.assertEqual(paddings.tolist(), [[0., 0., 0., 0., 0.]])

  def testStrToVocabTokenNoPadToMaxlen(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.str_to_vocab_tokens([
              'a b c d e',
              '<epsilon> <S> </S> <UNK>',
              'øut über ♣ 愤青 ←',
          ],
                                       append_eos=True,
                                       maxlen=10,
                                       pad_to_maxlen=False,
                                       vocab_filepath=vocab))
      self.assertEqual(
          token_ids.tolist(),
          [[1, 5, 6, 7, 8, 9], [1, 0, 1, 2, 3, 2], [1, 10, 11, 12, 13, 3]])
      self.assertEqual(
          target_ids.tolist(),
          [[5, 6, 7, 8, 9, 2], [0, 1, 2, 3, 2, 2], [10, 11, 12, 13, 3, 2]])
      self.assertEqual(paddings.tolist(),
                       [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1.],
                        [0., 0., 0., 0., 0., 0.]])

  def testStrToVocabTokenCustomDelimiter(self):
    custom_delimiter = '_'
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.str_to_vocab_tokens([custom_delimiter.join('abcde')],
                                       append_eos=True,
                                       maxlen=8,
                                       vocab_filepath=vocab,
                                       delimiter=custom_delimiter))
      self.assertEqual(token_ids.tolist(), [[1, 5, 6, 7, 8, 9, 2, 2]])
      self.assertEqual(target_ids.tolist(), [[5, 6, 7, 8, 9, 2, 2, 2]])
      self.assertEqual(paddings.tolist(), [[0., 0., 0., 0., 0., 0., 1., 1.]])

  def testStrToVocabTokenSplitToCharacters(self):
    custom_delimiter = ''
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False) as sess:
      token_ids, target_ids, paddings = sess.run(
          py_x_ops.str_to_vocab_tokens(['abcde'],
                                       append_eos=True,
                                       maxlen=8,
                                       vocab_filepath=vocab,
                                       delimiter=custom_delimiter))
      self.assertEqual(token_ids.tolist(), [[1, 5, 6, 7, 8, 9, 2, 2]])
      self.assertEqual(target_ids.tolist(), [[5, 6, 7, 8, 9, 2, 2, 2]])
      self.assertEqual(paddings.tolist(), [[0., 0., 0., 0., 0., 0., 1., 1.]])

  def testNgramIdToToken(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_ngrams.txt')
    with self.session(use_gpu=False):
      ngram_ids = [[14, 11, 6, 24, 7, 3, 13, 82, 2, 2],
                   [57, 3, 73, 17, 22, 9, 2, 2, 2, 2]]
      lengths = [8, 6]
      scripts = py_x_ops.ngram_id_to_token(
          ngram_ids, lengths, ngram_vocab_filepath=vocab)
      scripts_expected = ['pn?o"{twe', 'gh{rtlcr']
      self.assertEqual(scripts_expected, scripts.eval().tolist())

  def testNgramIdToTokenSeparator(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_ngrams.txt')
    with self.session(use_gpu=False):
      ngram_ids = [[14, 11, 6, 24, 7, 3, 13, 82, 2, 2],
                   [57, 3, 73, 17, 22, 9, 2, 2, 2, 2]]
      lengths = [8, 6]
      scripts = py_x_ops.ngram_id_to_token(
          ngram_ids, lengths, ngram_vocab_filepath=vocab, ngram_separator='.')
      scripts_expected = ['p.n.?.o.".{.t.we', 'gh.{.rt.l.c.r']
      self.assertEqual(scripts_expected, scripts.eval().tolist())

  def testBpeTokenization(self):
    word_vocab = test_helper.test_src_dir_path(
        'core/ops/testdata/bpe_words.vocab')
    code_vocab = test_helper.test_src_dir_path(
        'core/ops/testdata/bpe_codes.vocab')
    sentences = [
        'GIVE ME A PENNY', 'THEY LIVED ALONE', 'THEY GIVE ME A PENNY ALONE'
    ]
    expected_sentences = [
        'GIVE ME A PENNY </s> ',
        'THEY LIVED ALONE </s> ',
        'THEY GIVE ME A PENNY ',
    ]
    expected_token_ids = [
        [27, 9, 30, 14, 28, 14, 52, 11, 4, 6, 6, 10, 2, 2, 2],
        [16, 4, 10, 12, 9, 30, 24, 7, 12, 49, 14, 2, 2, 2, 2],
        [16, 4, 10, 27, 9, 30, 14, 28, 14, 52, 11, 4, 6, 6, 10],
    ]
    with self.session(use_gpu=False):
      label_tensor = tf.constant(sentences)
      _, token_ids, paddings = py_x_ops.bpe_words_to_ids(
          label_tensor, tokenization_filepath=word_vocab, maxlen=15)
      seq_lens = tf.cast(tf.reduce_sum(1 - paddings, axis=-1), tf.int32)

      target_string = py_x_ops.bpe_ids_to_words(
          token_ids, seq_lengths=seq_lens, vocab_filepath=code_vocab)
      self.assertEqual(expected_sentences, target_string.eval().tolist())
      self.assertEqual(expected_token_ids, token_ids.eval().tolist())


if __name__ == '__main__':
  tf.test.main()
