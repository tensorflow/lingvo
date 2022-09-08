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
"""Html handler customizations for simple_wer_v2."""

import re
from lingvo.tasks.asr.tools import simple_wer_v2


# pylint: disable=arguments-renamed
class ChainOfHtmlHandlers(simple_wer_v2.HtmlHandler):
  """A handler that is a chain of handlers.

  All Setup and Render functions are called in sequential order.
  """

  def __init__(self, html_handlers=None):
    assert html_handlers, 'Must provide list of handlers as input.'
    self._html_handlers = html_handlers

  def Setup(self, hypothesis, reference):
    for handler in self._html_handlers:
      handler.Setup(hypothesis, reference)

  def Render(self, **kwargs):
    return ''.join(
        [handler.Render(**kwargs) for handler in self._html_handlers])


def FindTags(hyp_words):
  """Find the tags in the hypothesis.

  Tags are words enclosed by angle-brackets, such as <tag>.
  Tags are meant to be visible and not affect WER.

  Args:
    hyp_words: List of words in the hypothesis sentence.

  Returns:
    A list of tags in the hypothesis. Each tag is a 2-element tuple. First
    element is the position, second is the string.
  """
  tags = []
  for pos, word in enumerate(hyp_words):
    if re.search(r'\<[\w_.]+\>', word):
      tags.append((pos - len(tags), word))
  return tags


class TagHtmlHandler(simple_wer_v2.HtmlHandler):
  """Handler to cache and add tags back to original positions in transcript."""

  def Setup(self, hypothesis, reference):
    # Cache tag information before they're removed for WER computation.
    self.tags = FindTags(hypothesis.split())

  def Render(self, pos_hyp=None, **kwargs):
    """Show tags in the output html.

    Args:
      pos_hyp: current word position in the hypothesis
      **kwargs: unused

    Returns:
      Tag strings
    """
    tag_strs = []
    while self.tags and pos_hyp == self.tags[-1][0]:
      _, tag_str = self.tags.pop()
      tag_strs.append(tag_str)
    if tag_strs:
      return ' '.join(tag_strs[::-1]) + ' '
    return ''


class NewlineHtmlHandler(simple_wer_v2.HtmlHandler):
  """Handler to insert newline into html at fixed ref word intervals.

  Useful for side-by-side comparisons of long transcripts.
  """

  def __init__(self, num_words_per_line=-1):
    """Specify the number of reference words to display on each line.

    Args:
      num_words_per_line: number of ref words on each line
    """
    self.num_words_per_line = num_words_per_line

  def Setup(self, hypothesis, reference):
    self.line_pos = 0

  def Render(self, err_type=None, **kwargs):
    """Set number of ref words to display per line.

    Args:
      err_type: error type
      **kwargs: unused

    Returns:
      Newline string when number of words reaches num_words_per_line
    """
    if err_type != 'ins' and self.num_words_per_line > 0:
      self.line_pos += 1
      if self.line_pos % self.num_words_per_line == 0:
        self.line_pos = 0
        return '<br> '
    return ''
