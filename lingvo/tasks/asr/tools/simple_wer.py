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
"""Stand-alone script to evalute the word error rate (WER) for ASR tasks.

Tensorflow and Lingvo are not required to run this script.

Example of Usage::

`python simple_wer.py file_hypothesis file_reference`

where `file_hypothesis` is the file name for hypothesis text and
`file_reference` is the file name for reference text.

Or you can use this file as a library, and call either of the following:

  - `ComputeWER(hyp, ref)`    compute WER for one pair of hypothesis/reference
  - `AverageWERs(hyps, refs)` average WER for a list of hypotheses/references

Note to evaluate the ASR, we consider the following pre-processing:

  - change transcripts to lower-case
  - remove punctuation: `" , . ! ? (  ) [ ]`
  - remove extra empty spaces
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys


def _ComputeEditDistance(hs, rs):
  """Compute edit distance between two list of strings.

  Args:
    hs: the list of words in the hypothesis sentence
    rs: the list of words in the reference sentence

  Returns:
    edit distance as an integer
  """
  dr, dh = len(rs) + 1, len(hs) + 1
  dists = [[]] * dr

  # initialization for dynamic programming
  for i in range(dr):
    dists[i] = [0] * dh
    for j in range(dh):
      if i == 0:
        dists[0][j] = j
      elif j == 0:
        dists[i][0] = i

  # do dynamic programming
  for i in range(1, dr):
    for j in range(1, dh):
      if rs[i - 1] == hs[j - 1]:
        dists[i][j] = dists[i - 1][j - 1]
      else:
        tmp0 = dists[i - 1][j - 1] + 1
        tmp1 = dists[i][j - 1] + 1
        tmp2 = dists[i - 1][j] + 1
        dists[i][j] = min(tmp0, tmp1, tmp2)

  return dists[-1][-1]


def _PreprocessTxtBeforeWER(txt):
  """Preprocess text before WER caculation."""

  # lower case, and remove \t and new line
  txt = re.sub(r'[\t\n]', ' ', txt.lower())

  # remove punctuation before space
  txt = re.sub(r'[,.\?!]+ ', ' ', txt)

  # remove punctuation before end
  txt = re.sub(r'[,.\?!]+$', ' ', txt)

  # remove punctuation after space
  txt = re.sub(r' [,.\?!]+', ' ', txt)

  # remove quotes, [, ], ( and )
  txt = re.sub(r'["\(\)\[\]]', '', txt)

  # remove extra space
  txt = re.sub(' +', ' ', txt.strip())

  return txt


def ComputeWER(hyp, ref):
  """Computes WER for ASR by ignoring diff of punctuation, space, captions.

  Args:
    hyp: Hypothesis string.
    ref: Reference string.

  Returns:
    num of errors, num of reference words
  """

  hyp = _PreprocessTxtBeforeWER(hyp)
  ref = _PreprocessTxtBeforeWER(ref)

  # compute num of word errors
  hs = hyp.split()
  rs = ref.split()
  d = _ComputeEditDistance(hs, rs)

  # num of words. For empty ref we set num = 1
  nr = max(len(rs), 1)
  return d, nr


def AverageWERs(hyps, refs, verbose=True):
  """Computes average WER from a list of references/hypotheses.

  Args:
    hyps: list of hypothesis strings.
    refs: list of reference strings.
    verbose: optional (default True)

  Returns:
    total num of errors, total num of words in refs
  """
  totale, totalw = 0, 0
  for hyp, ref in zip(hyps, refs):
    ei, ni = ComputeWER(hyp, ref)
    totale += ei
    totalw += ni

  if verbose:
    print('total error = %d, total word = %d, wer = %.2f' %
          (totale, totalw, totale * 100.0 / totalw))
  return totale, totalw


def main(argv):

  hyp = open(argv[1], 'r').read()
  ref = open(argv[2], 'r').read()

  ne, nw = ComputeWER(hyp, ref)
  print('num of error = %d, num of word = %d, wer = %.2f' %
        (ne, nw, ne * 100.0 / nw))


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("""
Example of Usage:

  python simple_wer.py file_hypothesis file_reference
  
  where file_hypothesis is the file name for hypothesis text
  and   file_reference  is the file name for reference text.

Or you can use this file as a library, and call either of the following
  - ComputeWER(hyp, ref)    to compute WER for one pair of hypothesis/reference
  - AverageWERs(hyps, refs) to average WER for a list of hypotheses/references
""")
    sys.exit(1)

  main(sys.argv)
