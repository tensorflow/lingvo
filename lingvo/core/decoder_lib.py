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

import pickle

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core.ops import record_pb2


def WriteKeyValuePairs(filename, key_value_pairs):
  """Writes `key_value_pairs` to `filename`."""
  with open(filename, 'wb') as f:
    pickle.dump(key_value_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def SerializeOutputs(nmap: py_utils.NestedMap) -> bytes:
  """Return a serialized representation of the contents of `nmap`.

  Args:
    nmap: A NestedMap of data to serialize.

  Returns:
    A serialized record_pb2.Record() of the contents of `nmap`.
  """
  record = record_pb2.Record()
  flat_nmap = nmap.FlattenItems()
  for key, value in flat_nmap:
    record.fields[key].CopyFrom(tf.make_tensor_proto(value))
  serialized = record.SerializeToString()
  return serialized
