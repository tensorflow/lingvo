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
"""Wrapper for pdb."""

import contextlib
import pdb
from absl import flags

flags.DEFINE_bool('pdb_on_exception', False,
                  'If True, enter pdb trace on fatal exceptions.')

set_trace = pdb.set_trace
post_mortem = pdb.post_mortem


@contextlib.contextmanager
def catch_post_mortem():
  try:
    yield
  except Exception as e:  # pylint: disable=broad-exception
    post_mortem()
    raise
