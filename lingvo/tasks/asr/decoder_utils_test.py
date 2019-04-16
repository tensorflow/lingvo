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
"""Tests for decoder utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import test_utils

from lingvo.tasks.asr import decoder_utils

FLAGS = tf.flags.FLAGS


class DecoderUtilsTokenizeTest(test_utils.TestCase):

  def testTokenize(self):
    s = "onetoken"
    self.assertEqual(["onetoken"], decoder_utils.Tokenize(s))

    s = "two tokens"
    self.assertEqual(["two", "tokens"], decoder_utils.Tokenize(s))

    s = " extra  spaces    are filtered "
    self.assertEqual(["extra", "spaces", "are", "filtered"],
                     decoder_utils.Tokenize(s))


class DecoderUtilsComputeWerTest(test_utils.TestCase):

  def testInvalidInputsExtraHyps(self):
    with self.session():
      with self.assertRaises(Exception):
        decoder_utils.ComputeWer(hyps=["one", "two"], refs=["one"]).eval()

  def testInvalidInputsExtraRefs(self):
    with self.session():
      with self.assertRaises(Exception):
        decoder_utils.ComputeWer(hyps=["one"], refs=["one", "two"]).eval()

  def testInvalidInputsWrongRank(self):
    with self.session():
      with self.assertRaises(Exception):
        decoder_utils.ComputeWer(
            hyps=[["one"], ["two"]], refs=[["one"], ["two"]]).eval()

  def testBasic(self):
    with self.session():
      self.assertAllEqual(
          decoder_utils.ComputeWer(hyps=["one"], refs=["one"]).eval(), [[0, 1]])
      self.assertAllEqual(
          decoder_utils.ComputeWer(hyps=["one two"], refs=["one two"]).eval(),
          [[0, 2]])

  def testMultiples(self):
    with self.session():
      wer = decoder_utils.ComputeWer(
          hyps=["one", "two pigs"], refs=["one", "three pink pigs"])
      self.assertAllEqual(wer.shape, [2, 2])
      self.assertAllEqual(wer.eval(), [[0, 1], [2, 3]])

  def testConsecutiveWhiteSpace(self):
    with self.session():
      wer = decoder_utils.ComputeWer(
          hyps=["one    two", "one two", "two     pigs"],
          refs=["one two", "one     two ", "three pink pigs"])
      self.assertAllEqual(wer.shape, [3, 2])
      self.assertAllEqual(wer.eval(), [[0, 2], [0, 2], [2, 3]])

  def testEmptyRefsAndHyps(self):
    with self.session():
      wer = decoder_utils.ComputeWer(
          hyps=["", "one two", ""], refs=["", "", "three four five"])
      self.assertAllEqual(wer.shape, [3, 2])
      self.assertAllEqual(wer.eval(), [[0, 0], [2, 0], [3, 3]])

  def testDifferencesInCaseAreCountedAsErrors(self):
    with self.session():
      wer = decoder_utils.ComputeWer(
          hyps=["ONE two", "one two"], refs=["one two", "ONE two"])
      self.assertAllEqual(wer.shape, [2, 2])
      self.assertAllEqual(wer.eval(), [[1, 2], [1, 2]])


class DecoderUtilsFilterTest(test_utils.TestCase):

  def testFilterEpsilon(self):
    s = "no epsilon"
    self.assertEqual(s, decoder_utils.FilterEpsilon(s))

    s = "<epsilon>epsilon tokens are<epsilon>removed<epsilon>"
    self.assertEqual("epsilon tokens are removed",
                     decoder_utils.FilterEpsilon(s))

  def testFilterNoise(self):
    s = "no noise"
    self.assertEqual(s, decoder_utils.FilterNoise(s))

    s = "<noise> noise tokens are <noise> removed <noise>"
    self.assertEqual("noise tokens are removed", decoder_utils.FilterNoise(s))


class DecoderUtilsEditDistanceTest(test_utils.TestCase):

  def testEditDistance1(self):
    ref = "a b c d e f g h"
    hyp = "a b c d e f g h"
    self.assertEqual((0, 0, 0, 0), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d e f g h"
    hyp = "a b d e f g h"
    self.assertEqual((0, 0, 1, 1), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d e f g h"
    hyp = "a b c i d e f g h"
    self.assertEqual((1, 0, 0, 1), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d e f g h"
    hyp = "a b c i e f g h"
    self.assertEqual((0, 1, 0, 1), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d e f g j h"
    hyp = "a b c i d e f g h"
    self.assertEqual((1, 0, 1, 2), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d e f g j h"
    hyp = "a b c i e f g h k"
    self.assertEqual((1, 1, 1, 3), decoder_utils.EditDistance(ref, hyp))

    ref = ""
    hyp = ""
    self.assertEqual((0, 0, 0, 0), decoder_utils.EditDistance(ref, hyp))
    ref = ""
    hyp = "a b c"
    self.assertEqual((3, 0, 0, 3), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d"
    hyp = ""
    self.assertEqual((0, 0, 4, 4), decoder_utils.EditDistance(ref, hyp))

  def testEditDistanceInIds(self):
    ref = [0, 1, 2, 3, 9]
    hyp = [0, 2, 3, 5, 6]
    self.assertEqual((1, 1, 1, 3), decoder_utils.EditDistanceInIds(ref, hyp))

  def testEditDistanceSkipsEmptyTokens(self):
    ref = "a b c d e   f g h"
    hyp = "a b c d e f g h"
    self.assertEqual((0, 0, 0, 0), decoder_utils.EditDistance(ref, hyp))

    ref = "a b c d e f g h"
    hyp = "a b c d e   f g h"
    self.assertEqual((0, 0, 0, 0), decoder_utils.EditDistance(ref, hyp))


if __name__ == "__main__":
  tf.test.main()
