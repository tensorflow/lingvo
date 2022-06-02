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

import string
import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.punctuator import input_generator


class InputGeneratorTest(test_utils.TestCase):

  def _CreatePunctuatorInputParams(self):
    p = input_generator.PunctuatorInput.Params()
    input_file = test_helper.test_src_dir_path('tasks/lm/testdata/lm1b_100.txt')
    p.tokenizer.vocab_filepath = test_helper.test_src_dir_path(
        'tasks/punctuator/params/brown_corpus_wpm.16000.vocab')
    p.tokenizer.vocab_size = 16000
    p.file_datasource.file_pattern = input_file
    p.source_max_length = 200
    p.target_max_length = 200
    p.bucket_upper_bound = [20, 40]
    p.bucket_batch_limit = [1, 1]
    return p

  def testBasic(self):
    p = self._CreatePunctuatorInputParams()
    with self.session(use_gpu=False):
      inp = input_generator.PunctuatorInput(p)
      # Runs a few steps.
      for _ in range(3):
        self.evaluate(inp.GetPreprocessedInputBatch())

  def testSourceTargetValues(self):
    max_length = 50
    p = self._CreatePunctuatorInputParams()
    with self.session(use_gpu=False):
      inp = input_generator.PunctuatorInput(p)

      fetched = py_utils.NestedMap(
          self.evaluate(inp.GetPreprocessedInputBatch()))
      source_ids = fetched.src.ids
      tgt_ids = fetched.tgt.ids
      tgt_labels = fetched.tgt.labels

      expected_ref = (
          b'Elk calling -- a skill that hunters perfected long ago to lure '
          b'game with the promise of a little romance -- is now its own sport .'
      )

      normalized_ref = expected_ref.lower().translate(
          None, string.punctuation.encode('utf-8'))
      normalized_ref = b' '.join(normalized_ref.split())
      _, expected_src_ids, _ = self.evaluate(
          inp.tokenizer.StringsToIds(
              tf.convert_to_tensor([normalized_ref]), max_length=max_length))
      expected_tgt_ids, expected_tgt_labels, _ = self.evaluate(
          inp.tokenizer.StringsToIds(
              tf.convert_to_tensor([expected_ref]), max_length=max_length))

      self.assertAllEqual(expected_src_ids[0], source_ids[0, :max_length])
      self.assertAllEqual(expected_tgt_ids[0], tgt_ids[0, :max_length])
      self.assertAllEqual(expected_tgt_labels[0], tgt_labels[0, :max_length])


if __name__ == '__main__':
  test_utils.main()
