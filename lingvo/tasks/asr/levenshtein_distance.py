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
"""Common utilities for ASR decoders."""
import copy
from typing import List


class ErrorStats:
  """Class to keep track of error counts."""

  def __init__(self, ins, dels, subs, tot):
    self.insertions, self.deletions, self.subs, self.total = (ins, dels, subs,
                                                              tot)

  def __repr__(self):
    return f'ErrorStats(ins={self.insertions}, dels={self.deletions}, subs={self.subs}, tot={self.total})'


def LevenshteinDistance(lst_ref: List[str], lst_hyp: List[str]) -> ErrorStats:
  """Computes Levenshtein edit distance between reference and hypotheses."""
  # temp sequence to remember error type and stats.
  e, cur_e = [], []
  for i in range(len(lst_ref) + 1):
    e.append(ErrorStats(0, i, 0, i))
    cur_e.append(ErrorStats(0, 0, 0, 0))

  for hyp_index in range(1, len(lst_hyp) + 1):
    cur_e[0] = copy.copy(e[0])
    cur_e[0].insertions += 1
    cur_e[0].total += 1

    for ref_index in range(1, len(lst_ref) + 1):
      ins_err = e[ref_index].total + 1
      del_err = cur_e[ref_index - 1].total + 1
      sub_err = e[ref_index - 1].total
      if lst_hyp[hyp_index - 1] != lst_ref[ref_index - 1]:
        sub_err += 1

      if sub_err < ins_err and sub_err < del_err:
        cur_e[ref_index] = copy.copy(e[ref_index - 1])
        if lst_hyp[hyp_index - 1] != lst_ref[ref_index - 1]:
          cur_e[ref_index].subs += 1
        cur_e[ref_index].total = sub_err
      elif del_err < ins_err:
        cur_e[ref_index] = copy.copy(cur_e[ref_index - 1])
        cur_e[ref_index].total = del_err
        cur_e[ref_index].deletions += 1
      else:
        cur_e[ref_index] = copy.copy(e[ref_index])
        cur_e[ref_index].total = ins_err
        cur_e[ref_index].insertions += 1

    for i in range(len(e)):
      e[i] = copy.copy(cur_e[i])

  return e[-1]
