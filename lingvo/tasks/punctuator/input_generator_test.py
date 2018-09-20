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
"""Tests for input generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import tensorflow as tf

from lingvo.core import py_utils
from lingvo.core import test_helper
from lingvo.tasks.punctuator import input_generator


class InputGeneratorTest(tf.test.TestCase):

  def _CreatePunctuatorInputParams(self):
    p = input_generator.PunctuatorInput.Params()
    input_file = 'text:' + test_helper.test_src_dir_path(
        'tasks/lm/testdata/lm1b_100.txt')
    p.tokenizer.token_vocab_filepath = test_helper.test_src_dir_path(
        'tasks/punctuator/testdata/test_vocab.txt')
    p.file_pattern = input_file
    p.file_random_seed = 314
    p.file_parallelism = 1
    p.source_max_length = 1000
    p.target_max_length = 1000
    p.bucket_upper_bound = [20, 40]
    p.bucket_batch_limit = [1, 1]
    return p

  def testBasic(self):
    p = self._CreatePunctuatorInputParams()
    with self.session(use_gpu=False) as sess:
      inp = input_generator.PunctuatorInput(p)
      # Runs a few steps.
      for _ in range(10):
        sess.run(inp.GetPreprocessedInputBatch())

  def testSourceTargetValues(self):
    max_length = 50
    p = self._CreatePunctuatorInputParams()
    with self.session(use_gpu=False) as sess:
      inp = input_generator.PunctuatorInput(p)
      tokenizer = inp.tokenizer_dict['default']

      fetched = py_utils.NestedMap(sess.run(inp.GetPreprocessedInputBatch()))
      source_ids = fetched.src.ids
      tgt_ids = fetched.tgt.ids
      tgt_labels = fetched.tgt.labels

      expected_ref = 'The internet is sort-of-40 this year .'

      # "the internet is sortof40 this year" - lower-case, no dashes, no dot.
      normalized_ref = expected_ref.lower().translate(None, string.punctuation)
      expected_src_ids, _, _ = sess.run(
          tokenizer.StringsToIds([normalized_ref], max_length=max_length))
      expected_tgt_ids, expected_tgt_labels, _ = sess.run(
          tokenizer.StringsToIds([expected_ref], max_length=max_length))

      self.assertAllEqual(expected_src_ids[0], source_ids[0, :max_length])
      self.assertAllEqual(expected_tgt_ids[0], tgt_ids[0, :max_length])
      self.assertAllEqual(expected_tgt_labels[0], tgt_labels[0, :max_length])


if __name__ == '__main__':
  tf.test.main()
