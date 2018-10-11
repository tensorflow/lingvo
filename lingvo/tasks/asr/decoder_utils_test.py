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

from lingvo.tasks.asr import decoder_utils

FLAGS = tf.flags.FLAGS


class DecoderUtilsTokenizeTest(tf.test.TestCase):

  def testTokenize(self):
    s = "onetoken"
    self.assertEqual(["onetoken"], decoder_utils.Tokenize(s))

    s = "two tokens"
    self.assertEqual(["two", "tokens"], decoder_utils.Tokenize(s))

    s = " extra  spaces    are filtered "
    self.assertEqual(["extra", "spaces", "are", "filtered"],
                     decoder_utils.Tokenize(s))


class DecoderUtilsComputeWerTest(tf.test.TestCase):

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


if __name__ == "__main__":
  tf.test.main()
