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
"""Tests for lm.input_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.lm import input_generator


class InputGeneratorTest(test_utils.TestCase):

  def _InputParams(self):
    p = input_generator.LmInput.Params()
    p.file_pattern = "text:" + test_helper.test_src_dir_path(
        "tasks/lm/testdata/lm1b_100.txt")
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p.bucket_upper_bound = [20]
    p.bucket_batch_limit = [2]
    p.target_max_length = 20
    return p

  def testLmInputGen(self):
    p = self._InputParams()

    with self.session(use_gpu=False) as sess:
      inp = p.cls(p)
      inp_batch = sess.run(inp.InputBatch())
      print(inp_batch)
      # pyformat: disable
      # pylint: disable=line-too-long
      expected_ids = [
          [1, 13, 24, 3, 23, 5, 13, 8, 3, 24, 12, 5, 24, 3, 24, 27, 19, 3, 19, 10],
          [1, 24, 12, 9, 3, 17, 5, 14, 19, 22, 13, 24, 29, 3, 27, 13, 16, 16, 3, 6],
      ]
      # pylint: enable=line-too-long
      # pyformat: enable
      self.assertEqual(expected_ids, inp_batch.ids.tolist())
      self.assertEqual([[1.0] * 20, [1.0] * 20], inp_batch.weights.tolist())


if __name__ == "__main__":
  tf.test.main()
