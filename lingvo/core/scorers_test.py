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
"""Tests for scorers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from lingvo.core import scorers
from lingvo.core import test_helper
from lingvo.core import test_utils


class BleuScorerTest(test_utils.TestCase):

  def testNGrams(self):
    words = 'a b c d e'.split(' ')
    self.assertEqual([['a'], ['b'], ['c'], ['d'], ['e']],
                     list(scorers.NGrams(words, 1)))

    self.assertEqual([['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']],
                     list(scorers.NGrams(words, 2)))

    self.assertEqual([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']],
                     list(scorers.NGrams(words, 3)))

  def testBleuScorerDocTest(self):
    scorer = scorers.BleuScorer(max_ngram=4)
    scorer.AddSentence('hyp matches ref str', 'hyp matches ref str')
    self.assertAlmostEqual((4/4 * 3/3 * 2/2 * 1/1) ** (1/4),
                           scorer.ComputeOverallScore())
    scorer.AddSentence('almost right', 'almost write')
    self.assertAlmostEqual((5/6 * 3/4 * 2/2 * 1/1) ** (1/4),
                           scorer.ComputeOverallScore())

  def testBleuScorerClipsExtraHypNGrams(self):
    scorer = scorers.BleuScorer(max_ngram=4)
    scorer.AddSentence('a b c d', 'a a b c d')
    self.assertAlmostEqual((4/5 * 3/4 * 2/3 * 1/2) ** (1/4),
                           scorer.ComputeOverallScore())

  def testBleuScorerSentencesShorterThanMaxNGram(self):
    scorer = scorers.BleuScorer(max_ngram=4)
    scorer.AddSentence('', '')
    self.assertAlmostEqual(0.0, scorer.ComputeOverallScore())
    scorer.AddSentence('a', 'a')
    self.assertAlmostEqual(1.0, scorer.ComputeOverallScore())
    scorer.AddSentence('a b', 'a b')
    self.assertAlmostEqual(1.0, scorer.ComputeOverallScore())
    scorer.AddSentence('a b c', 'a b c')
    self.assertAlmostEqual(1.0, scorer.ComputeOverallScore())
    scorer.AddSentence('a b c d', 'a b c d')
    self.assertAlmostEqual(1.0, scorer.ComputeOverallScore())

  def testBleuScorerBrevityPenalty(self):
    scorer = scorers.BleuScorer(max_ngram=4)
    scorer.AddSentence('1 2 3 4 5', '1 2 3 4 -2')
    # No penalty.
    expected_score = (4/5 * 3/4 * 2/3 * 1/2) ** (1/4)
    self.assertAlmostEqual(0.6687403, expected_score)
    self.assertAlmostEqual(expected_score, scorer.ComputeOverallScore())

    scorer = scorers.BleuScorer(max_ngram=4)
    scorer.AddSentence('1 2 3 4 5', '1 2 3 4')
    expected_score = math.exp(1 - 5/4) * (4/4 * 3/3 * 2/2 * 1/1) ** (1/4)
    self.assertAlmostEqual(0.7788008, expected_score)
    self.assertAlmostEqual(expected_score, scorer.ComputeOverallScore())

  def testBleuScorerMatchesCppImplementationOneExample(self):
    scorer = scorers.BleuScorer()
    scorer.AddSentence(
        'It is obvious that this will have a certain influence .',
        'It is clear that this will have a certain influence .')
    self.assertAlmostEqual(0.74194467, scorer.ComputeOverallScore())

  def testBleuScorerMatchesCppImplementation(self):
    filename = test_helper.test_src_dir_path('core/ops/testdata/wmt/sm18.txt')
    scorer = scorers.BleuScorer()
    with open(filename, 'rb') as fp:
      for line in fp:
        hyp, ref = line[:-1].split('\t')
        scorer.AddSentence(ref, hyp)
    self.assertAlmostEqual(0.313776, scorer.ComputeOverallScore(), places=5)


if __name__ == '__main__':
  tf.test.main()
