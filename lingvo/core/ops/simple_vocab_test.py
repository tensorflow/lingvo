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
"""Tests for simple_vocab."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import test_utils
from lingvo.core.ops import py_x_ops


class VocabOpsTest(test_utils.TestCase):

  def testVocabTokenToId(self):
    with self.session(use_gpu=False):
      vocab = [
          '<S>',
          '</S>',
          '<UNK>',
          '<epsilon>',
          'a',
          'b c d e',
          'øut',
          'über',
          '♣',
          '愤青',
          '←',
      ]
      self.assertEqual(0, py_x_ops.vocab_token_to_id('<S>', vocab=vocab).eval())
      self.assertEqual(4, py_x_ops.vocab_token_to_id('a', vocab=vocab).eval())
      self.assertAllEqual([5, 8],
                          py_x_ops.vocab_token_to_id(['b c d e', '♣'],
                                                     vocab=vocab).eval())
      self.assertEqual(
          2,
          py_x_ops.vocab_token_to_id('unknown', vocab=vocab).eval())

  def testVocabTokenToIdLoadId(self):
    with self.session(use_gpu=False):
      vocab = [
          '<S>	3',
          '</S>	5',
          '<UNK>	7',
          '<epsilon>	9',
          'a	2',
          'b c d e	4',
          'øut	8',
          'über	10',
          '♣	-1',
          '愤青	-3',
          '←	-5',
      ]
      self.assertEqual(
          3,
          py_x_ops.vocab_token_to_id(
              '<S>', vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          2,
          py_x_ops.vocab_token_to_id(
              'a', vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertAllEqual([4, -1],
                          py_x_ops.vocab_token_to_id(
                              ['b c d e', '♣'],
                              vocab=vocab,
                              load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          7,
          py_x_ops.vocab_token_to_id(
              'unknown', vocab=vocab, load_token_ids_from_vocab=True).eval())

  def testVocabIdToToken(self):
    with self.session(use_gpu=False):
      vocab = [
          '<S>',
          '</S>',
          '<UNK>',
          '<epsilon>',
          'a',
          'b c d e',
          'øut',
          'über',
          '♣',
          '愤青',
          '←',
      ]
      self.assertEqual('<S>', py_x_ops.vocab_id_to_token(0, vocab=vocab).eval())
      self.assertEqual('a', py_x_ops.vocab_id_to_token(4, vocab=vocab).eval())
      self.assertAllEqual(['b c d e', '♣'],
                          py_x_ops.vocab_id_to_token([5, 8],
                                                     vocab=vocab).eval())
      self.assertEqual('<UNK>',
                       py_x_ops.vocab_id_to_token(2, vocab=vocab).eval())
      self.assertEqual('<UNK>',
                       py_x_ops.vocab_id_to_token(-1, vocab=vocab).eval())
      self.assertEqual('<UNK>',
                       py_x_ops.vocab_id_to_token(11, vocab=vocab).eval())

  def testVocabIdToTokenLoadId(self):
    with self.session(use_gpu=False):
      vocab = [
          '<S>	3',
          '</S>	5',
          '<UNK>	7',
          '<epsilon>	9',
          'a	2',
          'b c d e	4',
          'øut	8',
          'über	10',
          '♣	-1',
          '愤青	-3',
          '←	-5',
      ]
      self.assertEqual(
          '<S>',
          py_x_ops.vocab_id_to_token(
              3, vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          'a',
          py_x_ops.vocab_id_to_token(
              2, vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertAllEqual(['b c d e', '♣'],
                          py_x_ops.vocab_id_to_token(
                              [4, -1],
                              vocab=vocab,
                              load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          '<UNK>',
          py_x_ops.vocab_id_to_token(
              7, vocab=vocab, load_token_ids_from_vocab=True).eval())
      self.assertEqual(
          '<UNK>',
          py_x_ops.vocab_id_to_token(
              0, vocab=vocab, load_token_ids_from_vocab=True).eval())

  def testTokenInVocab(self):
    with self.session(use_gpu=False):
      vocab = [
          '<S>',
          '</S>',
          '<UNK>',
          '<epsilon>',
          'a',
          'b c d e',
          'øut',
          'über',
          '♣',
          '愤青',
          '←',
      ]
      self.assertTrue(py_x_ops.token_in_vocab('a', vocab=vocab).eval())
      self.assertTrue(py_x_ops.token_in_vocab('<UNK>', vocab=vocab).eval())
      self.assertTrue(
          py_x_ops.token_in_vocab(['b c d e', '♣'], vocab=vocab).eval().all())
      self.assertFalse(py_x_ops.token_in_vocab('unknown', vocab=vocab).eval())


if __name__ == '__main__':
  tf.test.main()
