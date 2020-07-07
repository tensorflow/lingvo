# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tokenizer for use for the MLPerf transformer benchmark."""

from lingvo.core import ops
from lingvo.core import tokenizers


class MlPerfTokenizer(tokenizers.BaseTokenizer):
  """Id->String only for MLPerf decoding."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define("vocab_filepath", None, "Specifies a filepath to the vocab.")
    return p

  def IdsToStrings(self, ids, lens):
    p = self.params
    return ops.ml_perf_subword_id_to_string(
        ids, lens, vocab_filepath=p.vocab_filepath)
