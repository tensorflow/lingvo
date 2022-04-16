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
"""Tests for quantizer_objectives."""

from absl.testing import absltest
from lingvo.jax import test_utils
from lingvo.jax.layers import quantizer_objectives
import numpy as np


class CodebookObjectivesTest(test_utils.TestCase):
  codes = np.array([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [0, 0]],
                    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [0, 0]]])

  paddings = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])
  entropy = 1.609
  pplx = 5.000
  num_classes = 11

  def test_batch_pplx_entropy_from_codes(self):
    pplx, entropy, _ = quantizer_objectives.batch_pplx_entropy_from_codes(
        codes=self.codes, num_classes=self.num_classes, paddings=self.paddings)

    self.assertAlmostEqual(
        pplx, self.pplx, delta=1e-3, msg='PPLX is not the same')
    self.assertAlmostEqual(
        entropy, self.entropy, delta=1e-3, msg='Entropy is not the same')


if __name__ == '__main__':
  absltest.main()
