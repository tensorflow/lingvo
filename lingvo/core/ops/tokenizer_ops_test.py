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
# ==============================================================================
"""Tests for tokenizer_ops."""

from lingvo import compat as tf
from lingvo.core import ops
from lingvo.core import test_helper
from lingvo.core import test_utils
import six


class TokenizerOpsTest(test_utils.TestCase):

  def testLabelsToTokenId(self):
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.ascii_to_token_id([
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
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.ascii_to_token_id([
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
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.ascii_to_token_id([
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
    with self.session(use_gpu=False):
      token_ids = [[12, 9, 16, 16, 19, 2, 2, 2, 2,
                    2], [23, 13, 22, 73, 2, 2, 2, 2, 2,
                         2], [27, 12, 5, 24, 3, 5, 3, 0, 3,
                              2], [5, 17, 9, 22, 13, 7, 5, 32, 23, 2],
                   [4, 3, 9, 5, 22, 16, 29, 2, 2,
                    2], [40, 34, 39, 39, 3, 5, 17, 2, 2,
                         2], [52, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
      seq_lens = [5, 4, 9, 9, 7, 7, 1]
      tokens = self.evaluate(ops.id_to_ascii(token_ids, seq_lens))

    self.assertEqual(tokens.tolist(), [
        b'hello', b'sir<epsilon>', b'what a <unk> ', b"america's",
        b'<noise> early', b'1:00 am', b'%'
    ])

  def testStrToVocabToken(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.str_to_vocab_tokens([
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
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.str_to_vocab_tokens([
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
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.str_to_vocab_tokens(['a b c d e ' * 1000],
                                  append_eos=True,
                                  maxlen=5,
                                  vocab_filepath=vocab))
      self.assertEqual(token_ids.tolist(), [[1, 5, 6, 7, 8]])
      self.assertEqual(target_ids.tolist(), [[5, 6, 7, 8, 9]])
      self.assertEqual(paddings.tolist(), [[0., 0., 0., 0., 0.]])

  def testStrToVocabTokenNoPadToMaxlen(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_vocab.txt')
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.str_to_vocab_tokens([
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
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.str_to_vocab_tokens([custom_delimiter.join('abcde')],
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
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          ops.str_to_vocab_tokens(['abcde'],
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
      scripts = ops.ngram_id_to_token(
          ngram_ids, lengths, ngram_vocab_filepath=vocab)
      scripts_expected = [b'pn?o"{twe', b'gh{rtlcr']
      self.assertEqual(scripts_expected, scripts.eval().tolist())

  def testMlPerf(self):
    tf.logging.info(dir(ops))
    vocab = test_helper.test_src_dir_path(
        'core/ops/testdata/mlperf.ende.subwords.vocab')
    with self.session(use_gpu=False):
      lengths = [26, 33]
      inputs = [[
          264, 4, 288, 189, 4138, 30725, 14, 1461, 2761, 243, 28, 2692, 9, 1679,
          2, 1218, 9, 4, 15190, 7, 427, 497, 7, 147, 3, 1, 0, 0, 0, 0, 0, 0, 0
      ],
                [
                    275, 52, 11, 1434, 215, 30372, 1546, 6997, 6495, 291, 802,
                    10, 9986, 6, 10, 1857, 12726, 13624, 2, 53, 18, 87, 15934,
                    5618, 12321, 127, 1565, 4209, 885, 777, 335, 3, 1
                ]]
      expected = [
          six.ensure_binary(
              'If the market “misbehaves,” farmers could be reduced to poverty,'
              ' leading to the neglect of large areas of Europe.<EOS>'),
          six.ensure_binary(
              'Wenn sich der Markt „daneben benimmt“ könnten die Bauern in die '
              'Armut abgleiten, was zu einer Vernachlässigung großer Teile '
              'Europas führen würde.<EOS>')
      ]
      decode = ops.ml_perf_subword_id_to_string(
          inputs, lengths, vocab_filepath=vocab)
      self.assertEqual(expected, decode.eval().tolist())

  def testNgramIdToTokenSeparator(self):
    vocab = test_helper.test_src_dir_path('core/ops/testdata/test_ngrams.txt')
    with self.session(use_gpu=False):
      ngram_ids = [[14, 11, 6, 24, 7, 3, 13, 82, 2, 2],
                   [57, 3, 73, 17, 22, 9, 2, 2, 2, 2]]
      lengths = [8, 6]
      scripts = ops.ngram_id_to_token(
          ngram_ids, lengths, ngram_vocab_filepath=vocab, ngram_separator='.')
      scripts_expected = [b'p.n.?.o.".{.t.we', b'gh.{.rt.l.c.r']
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
        b'GIVE ME A PENNY </s> ',
        b'THEY LIVED ALONE </s> ',
        b'THEY GIVE ME A PENNY ',
    ]
    expected_token_ids = [
        [27, 9, 30, 14, 28, 14, 52, 11, 4, 6, 6, 10, 2, 2, 2],
        [16, 4, 10, 12, 9, 30, 24, 7, 12, 49, 14, 2, 2, 2, 2],
        [16, 4, 10, 27, 9, 30, 14, 28, 14, 52, 11, 4, 6, 6, 10],
    ]
    with self.session(use_gpu=False):
      label_tensor = tf.constant(sentences)
      _, token_ids, paddings = ops.bpe_words_to_ids(
          label_tensor, tokenization_filepath=word_vocab, maxlen=15)
      seq_lens = tf.cast(
          tf.round(tf.reduce_sum(1 - paddings, axis=-1)), tf.int32)

      target_string = ops.bpe_ids_to_words(
          token_ids, seq_lengths=seq_lens, vocab_filepath=code_vocab)
      self.assertEqual(expected_sentences, target_string.eval().tolist())
      self.assertEqual(expected_token_ids, token_ids.eval().tolist())


if __name__ == '__main__':
  tf.test.main()
