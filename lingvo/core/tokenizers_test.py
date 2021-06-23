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
"""Tests for tokenizers."""

import lingvo.compat as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.core import tokenizers


class SentencePieceTokenizerTest(test_utils.TestCase):

  def _Params(self):
    p = tokenizers.SentencePieceTokenizer.Params()
    p.spm_model = test_helper.test_src_dir_path('core/testdata/en-1k.spm.model')
    p.vocab_size = 1024
    return p

  def testStringsToIds(self):
    p = self._Params()
    tokenizer = p.Instantiate()

    strs = ['Hello world!', 'why', '']

    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length=None)
    with self.session() as sess:
      ids, labels, paddings = sess.run([ids, labels, paddings])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    self.assertAllEqual([[136, 263,  36, 779, 185,   2],
                         [109, 534,   2,   0,   0,   0],
                         [  2,   0,   0,   0,   0,   0]],
                        labels)
    self.assertAllEqual([[  1, 136, 263,  36, 779, 185],
                         [  1, 109, 534,   2,   0,   0],
                         [  1,   2,   0,   0,   0,   0]],
                        ids)
    self.assertAllEqual([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1]],
                        paddings)
    # pylint: enable=bad-whitespace
    # pyformat: enable

  def testStringsToIds_MaxLength(self):
    p = self._Params()
    tokenizer = p.Instantiate()

    strs = ['Hello world!', 'why', '']

    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length=4)
    with self.session() as sess:
      ids, labels, paddings = sess.run([ids, labels, paddings])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    self.assertAllEqual([[136, 263,  36, 779],
                         [109, 534,   2,   0],
                         [  2,   0,   0,   0]],
                        labels)
    self.assertAllEqual([[  1, 136, 263,  36],
                         [  1, 109, 534,   2],
                         [  1,   2,   0,   0]],
                        ids)
    self.assertAllEqual([[0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 1, 1, 1]],
                        paddings)
    # pylint: enable=bad-whitespace
    # pyformat: enable

  def testIdsToStrings(self):
    p = self._Params()
    tokenizer = p.Instantiate()

    ids = [[136, 263, 36, 779, 185, 2], [109, 534, 2, 0, 0, 0],
           [2, 0, 0, 0, 0, 0]]
    lens = [6, 3, 1]

    strs = tokenizer.IdsToStrings(ids, lens)
    with self.session() as sess:
      strs = sess.run(strs)

    self.assertAllEqual(['Hello world!', 'why', ''], strs.astype(str))


if __name__ == '__main__':
  tf.test.main()
