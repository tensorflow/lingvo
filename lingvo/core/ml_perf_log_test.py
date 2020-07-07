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
"""Tests for lingvo.core.tshape."""

import json
import lingvo.compat as tf
from lingvo.core import ml_perf_log as mlp_log


class TestMLPerfLog:
  """Test mlperf log."""

  def testFormat(self):
    msg = mlp_log.mlperf_format('foo_key', {'whiz': 'bang'})
    parts = msg.split()
    assert parts[0] == ':::MLL'
    assert float(parts[1]) > 10
    assert parts[2] == 'foo_key:'
    j = json.loads(' '.join(parts[3:]))
    assert j['value'] == {'whiz': 'bang'}
    assert j['metadata']['lineno'] == 21
    assert 'test_mlp_log' in j['metadata']['file']


if __name__ == '__main__':
  tf.test.main()
