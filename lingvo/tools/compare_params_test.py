# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for compare_params."""

from lingvo import compat as tf
from lingvo.core import hyperparams
from lingvo.core import test_utils
from lingvo.tools import compare_params


class CompareParamsTest(test_utils.TestCase):

  def testBasic(self):
    h1 = hyperparams.Params()
    h1.Define('a', 1, '')
    h1.Define('b', 2, '')

    h2 = hyperparams.Params()
    h2.Define('a', 3, '')  # different value.
    h2.Define('c', 2, '')

    d1, d2, d3 = compare_params.hyperparams_text_diff(h1.ToText(), h2.ToText())
    self.assertEqual(d1, ['b'])
    self.assertEqual(d2, ['c'])
    self.assertEqual(d3, {'a': ('1', '3')})

    # Exercise print function
    compare_params.print_hyperparams_text_diff('h1', 'h2', d1, d2, d3)


if __name__ == '__main__':
  tf.test.main()
