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
`python simple_wer.py file_hypothesis file_reference diagnosis_html`


where `file_hypothesis` is the file name for hypothesis text and
`file_reference` is the file name for reference text.
`diagnosis_html` (optional) is the html filename to diagnose the errors.

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


def ComputeEditDistanceMatrix(hs, rs):
  """Compute edit distance between two list of strings.

  Args:
    hs: the list of words in the hypothesis sentence
    rs: the list of words in the reference sentence

  Returns:
    Edit distance matrix (in the format of list of lists), where the first
    index is the reference and the second index is the hypothesis.
  """
  dr, dh = len(rs) + 1, len(hs) + 1
  dists = [[]] * dr

  # Initialization.
  for i in range(dr):
    dists[i] = [0] * dh
    for j in range(dh):
      if i == 0:
        dists[0][j] = j
      elif j == 0:
        dists[i][0] = i

  # Do dynamic programming.
  for i in range(1, dr):
    for j in range(1, dh):
      if rs[i - 1] == hs[j - 1]:
        dists[i][j] = dists[i - 1][j - 1]
      else:
        tmp0 = dists[i - 1][j - 1] + 1
        tmp1 = dists[i][j - 1] + 1
        tmp2 = dists[i - 1][j] + 1
        dists[i][j] = min(tmp0, tmp1, tmp2)

  return dists


def PreprocessTxtBeforeWER(txt):
  """Preprocess text before WER caculation."""

  # Lowercase, remove \t and new line.
  txt = re.sub(r'[\t\n]', ' ', txt.lower())

  # Remove punctuation before space.
  txt = re.sub(r'[,.\?!]+ ', ' ', txt)

  # Remove punctuation before end.
  txt = re.sub(r'[,.\?!]+$', ' ', txt)

  # Remove punctuation after space.
  txt = re.sub(r' [,.\?!]+', ' ', txt)

  # Remove quotes, [, ], ( and ).
  txt = re.sub(r'["\(\)\[\]]', '', txt)

  # Remove extra space.
  txt = re.sub(' +', ' ', txt.strip())

  return txt


def _GenerateAlignedHtml(hyp, ref, err_type):
  """Generate a html element to highlight the difference between hyp and ref.

  Args:
    hyp: Hypothesis string.
    ref: Reference string.
    err_type: one of 'none', 'sub', 'del', 'ins'.

  Returns:
    a html string with
      - error of hyp shown in "(hyp)"
      - error of ref shown in "<del>ref</def>"
      - all errors highlighted with yellow background
  """

  highlighted_html = ''
  if err_type == 'none':
    highlighted_html += '%s ' % hyp

  elif err_type == 'sub':
    highlighted_html += """<span style="background-color: yellow">
        '<del>%s</del>(%s) </span>""" % (hyp, ref)

  elif err_type == 'del':
    highlighted_html += """<span style="background-color: yellow">
        '<del>%s</del></span>""" % (
            hyp)

  elif err_type == 'ins':
    highlighted_html += """<span style="background-color: yellow">
        '(%s) </span>""" % (
            ref)

  else:
    raise ValueError('unknown err_type ' + err_type)

  return highlighted_html


def _GenerateSummaryFromErrs(nref, errs):
  """Generate strings to summarize word errors.

  Args:
    nref: integer of total words in references
    errs: dict of three types of errors. e.g. {'sub':10, 'ins': 15, 'del': 3}

  Returns:
    str1: string summarizing total error, total word, WER,
    str2: string breaking down three errors: deleting, insertion, substitute
  """

  total_error = sum(errs.values())
  str_sum = 'total error = %d, total word = %d, wer = %.2f%%' % (
      total_error, nref, total_error * 100.0 / nref)

  str_details = 'Error breakdown: del = %.2f%%, ins=%.2f%%, sub=%.2f%%' % (
      errs['del'] * 100.0 / nref, errs['ins'] * 100.0 / nref,
      errs['sub'] * 100.0 / nref)

  return str_sum, str_details


def ComputeWER(hyp, ref, diagnosis=False):
  """Computes WER for ASR by ignoring diff of punctuation, space, captions.

  Args:
    hyp: Hypothesis string.
    ref: Reference string.
    diagnosis (optional): whether to generate diagnosis str (in html format)

  Returns:
    dict of three types of errors. e.g. {'sub':0, 'ins': 0, 'del': 0}
    num of reference words, integer
    aligned html string for diagnois (empty if diagnosis = False)
  """

  hyp = PreprocessTxtBeforeWER(hyp)
  ref = PreprocessTxtBeforeWER(ref)

  # Compute edit distance.
  hs = hyp.split()
  rs = ref.split()
  distmat = ComputeEditDistanceMatrix(hs, rs)

  # Back trace, to distinguish different errors: insert, deletion, substitution.
  ih, ir = len(hs), len(rs)
  errs = {'sub': 0, 'ins': 0, 'del': 0}
  aligned_html = ''
  while ih > 0 or ir > 0:
    err_type = ''

    # Distinguish error type by back tracking
    if ir == 0:
      err_type = 'ins'
    elif ih == 0:
      err_type = 'del'
    else:
      if hs[ih - 1] == rs[ir - 1]:  # correct
        err_type = 'none'
      elif distmat[ir][ih] == distmat[ir - 1][ih - 1] + 1:  # substitute
        err_type = 'sub'
      elif distmat[ir][ih] == distmat[ir - 1][ih] + 1:  # deletion
        err_type = 'del'
      elif distmat[ir][ih] == distmat[ir][ih - 1] + 1:  # insert
        err_type = 'ins'
      else:
        raise ValueError('fail to parse edit distance matrix')

    # Generate aligned_html
    if diagnosis:
      aligned_html += _GenerateAlignedHtml(hs[ih - 1], rs[ir - 1], err_type)

    # If no error, go to previous ref and hyp.
    if err_type == 'none':
      ih, ir = ih - 1, ir - 1
      continue

    # Update error.
    errs[err_type] += 1

    # Adjust position of ref and hyp.
    if err_type == 'del':
      ir = ir - 1
    elif err_type == 'ins':
      ih = ih - 1
    else:  # err_type == 'sub'
      ih, ir = ih - 1, ir - 1

  assert distmat[-1][-1] == sum(errs.values())

  # Num of words. For empty ref we set num = 1.
  nref = max(len(rs), 1)

  if aligned_html:
    str1, str2 = _GenerateSummaryFromErrs(nref, errs)
    aligned_html = str1 + ' (' + str2 + ')' + '<br>' + aligned_html

  return errs, nref, aligned_html


def AverageWERs(hyps, refs, verbose=True, diagnosis=False):
  """Computes average WER from a list of references/hypotheses.

  Args:
    hyps: list of hypothesis strings.
    refs: list of reference strings.
    verbose: optional (default True)
    diagnosis (optional): whether to generate list of diagnosis html

  Returns:
    dict of three types of errors. e.g. {'sub':0, 'ins': 0, 'del': 0}
    num of reference words, integer
    list of aligned html string for diagnosis (empty if diagnosis = False)

  """
  totalw = 0
  total_errs = {'sub': 0, 'ins': 0, 'del': 0}
  aligned_html_list = []

  for hyp, ref in zip(hyps, refs):
    errs_i, nref_i, diag_str = ComputeWER(hyp, ref, diagnosis)
    if diagnosis:
      aligned_html_list += [diag_str]

    totalw += nref_i
    total_errs['sub'] += errs_i['sub']
    total_errs['ins'] += errs_i['ins']
    total_errs['del'] += errs_i['del']

  str_summary, str_details = _GenerateSummaryFromErrs(totalw, total_errs)

  if diagnosis:
    str_overall = 'Overall: ' + str_summary + '(' + str_details + ')'
    aligned_html_list = [str_overall] + aligned_html_list

  if verbose:
    print(str_summary)
    print(str_details)

  return total_errs, totalw


def main(argv):

  hyp = open(argv[1], 'r').read()
  ref = open(argv[2], 'r').read()
  if len(argv) == 4:
    diagnosis = True
    fn_output = argv[3]
  else:
    diagnosis = False
    fn_output = None

  errs, nref, aligned_html = ComputeWER(hyp, ref, diagnosis)
  str_summary, str_details = _GenerateSummaryFromErrs(nref, errs)
  print(str_summary)
  print(str_details)

  if fn_output:
    with open(fn_output, 'wt') as fp:
      fp.write('<body><html>')
      fp.write('<div>%s</div>' % aligned_html)
      fp.write('</body></html>')


if __name__ == '__main__':
  if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("""
Example of Usage:

  python simple_wer.py file_hypothesis file_reference
or
  python simple_wer.py file_hypothesis file_reference diagnosis_html

  where file_hypothesis is the file name for hypothesis text
        file_reference  is the file name for reference text.
        diagnosis_html (optional) is the html filename to diagnose the errors.

Or you can use this file as a library, and call either of the following
  - ComputeWER(hyp, ref)    to compute WER for one pair of hypothesis/reference
  - AverageWERs(hyps, refs) to average WER for a list of hypotheses/references
""")
    sys.exit(1)

  main(sys.argv)
