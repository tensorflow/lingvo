# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for score_functions."""

from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.milan import score_functions


class ScoreFunctionsTest(test_utils.TestCase):

  def testDotProductScoreFunction(self):
    dot_product = score_functions.DotProductScoreFunction.Params().Instantiate()
    vector_a = tf.zeros([4] + [299], dtype=tf.float32)
    vector_b = tf.zeros([4] + [299], dtype=tf.float32)
    result = dot_product(vector_a, vector_b)

    self.assertEqual([4, 4], result.shape)


if __name__ == '__main__':
  tf.test.main()
