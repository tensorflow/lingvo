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
"""Tests for WPM tokenizer (which is not in ops test)."""

import lingvo.compat as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.core import tokenizers
import numpy as np


class WpmTokenizerTest(test_utils.TestCase):

  def testStringsTokenIds(self):
    p = tokenizers.WpmTokenizer.Params()
    p.vocab_filepath = test_helper.test_src_dir_path('tasks/mt/wpm-ende.voc')
    p.vocab_size = 32000
    wpm_tokenizer = p.Instantiate()
    with self.session(use_gpu=False):
      token_ids, target_ids, paddings = self.evaluate(
          wpm_tokenizer.StringsToIds(
              tf.constant(['would that it were so simple', 'this is it', ''],
                          dtype=tf.string), 6, True))
    self.assertAllEqual(
        token_ids, [[1, 926, 601, 560, 1273, 721], [1, 647, 470, 560, 2, 2],
                    [1, 2, 2, 2, 2, 2]])
    self.assertAllEqual(target_ids,
                        [[926, 601, 560, 1273, 721, 5490],
                         [647, 470, 560, 2, 2, 2], [2, 2, 2, 2, 2, 2]])
    self.assertAllEqual(paddings,
                        [[0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 1.],
                         [0., 1., 1., 1., 1., 1.]])

  def testIdsToStrings(self):
    p = tokenizers.WpmTokenizer.Params()
    p.vocab_filepath = test_helper.test_src_dir_path('tasks/mt/wpm-ende.voc')
    p.vocab_size = 32000
    wpm_tokenizer = p.Instantiate()
    with self.session(use_gpu=False):
      ref = tf.constant([
          'would that it were so simple',
          'this is it',
          '',
      ])
      _, target_ids, paddings = self.evaluate(
          wpm_tokenizer.StringsToIds(ref, 100, True))
      lens = np.argmax(paddings > 0.0, axis=1) - 1
      found = self.evaluate(wpm_tokenizer.IdsToStrings(target_ids, lens))
      self.assertAllEqual(ref, found)


if __name__ == '__main__':
  tf.test.main()
