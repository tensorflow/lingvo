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
"""Tests for lm.input_generator."""

import lingvo.compat as tf
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

    with self.session(use_gpu=False):
      inp = p.Instantiate()
      inp_batch = self.evaluate(inp.GetPreprocessedInputBatch())
      print(inp_batch)
      expected_ids = [
          [1, 8, 19, 18, 3, 32, 24, 3, 35, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          [
              1, 27, 12, 29, 3, 8, 19, 9, 23, 3, 9, 26, 9, 22, 29, 24, 12, 13,
              18, 11
          ],
      ]
      self.assertEqual(expected_ids, inp_batch.ids.tolist())
      self.assertEqual([[1.0] * 9 + [0.0] * 11, [1.0] * 20],
                       inp_batch.weights.tolist())


if __name__ == "__main__":
  tf.test.main()
