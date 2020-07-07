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
"""Tests for metrics."""

import lingvo.compat as tf
from lingvo.core import ml_perf_bleu_metric
from lingvo.core import test_utils


class MlPerfMetricsTest(test_utils.TestCase):

  def testMlPerfBleuMetric(self):
    m = ml_perf_bleu_metric.MlPerfBleuMetric()
    m.Update(u"a b a z", u"a b a c")
    m.Update(u"y f g d k l m", u"e f \u2028 d")
    self.assertAllClose(0.2638, m.value, atol=1e-03)


if __name__ == "__main__":
  tf.test.main()
