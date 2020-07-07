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
"""Tests for summary_utils."""

from lingvo import compat as tf
from lingvo.core import cluster_factory
from lingvo.core import summary_utils
from lingvo.core import test_utils


class SummaryUtilsTest(test_utils.TestCase):

  def testStatsCounter(self):
    with self.session():
      with cluster_factory.ForTestingWorker(add_summary=True):
        foo = summary_utils.StatsCounter('foo')
        val = foo.Value()
        inc = foo.IncBy(100)

      self.evaluate(tf.global_variables_initializer())
      self.assertAllEqual(0, val.eval())
      self.assertAllEqual(100, self.evaluate(inc))
      self.assertAllEqual(100, val.eval())
      self.assertAllEqual([100, 200], self.evaluate([val, inc]))
      self.assertAllEqual([200, 300], self.evaluate([val, inc]))
      summary = tf.Summary.FromString(self.evaluate(tf.summary.merge_all()))
      self.assertTrue(any('foo' in v.tag for v in summary.value))


if __name__ == '__main__':
  tf.test.main()
