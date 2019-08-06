# Lint as: python2, python3
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
"""Helpers for the decoding phase of jobs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle


def WriteKeyValuePairs(filename, key_value_pairs):
  """Writes `key_value_pairs` to `filename`."""
  with open(filename, 'wb') as f:
    pickle.dump(key_value_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
