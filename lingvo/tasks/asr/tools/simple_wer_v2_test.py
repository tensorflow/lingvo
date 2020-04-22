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
# ==============================================================================
"""Tests for simple_wer_v2."""

import lingvo.compat as tf
from lingvo.core import test_utils
from lingvo.tasks.asr.tools import simple_wer_v2 as simple_wer


class SimpleWerTest(test_utils.TestCase):

  def testTxtPreprocess(self):
    txt = 'abcd [ABC] (fg) ???   OK! '
    txt_preprocessed = simple_wer.TxtPreprocess(txt)
    self.assertEqual(txt_preprocessed, 'abcd abc fg ok')

  def testRemoveCommentTxtPreprocess(self):
    txt = 'abcd [ABC] (fg) ???   OK! '
    txt_preprocessed = simple_wer.RemoveCommentTxtPreprocess(txt)
    self.assertEqual(txt_preprocessed, 'abcd fg ok')

  def testHighlightAlignedHtml(self):
    hyp = 'thank you'
    ref = 'thank you'
    html = simple_wer.HighlightAlignedHtml(hyp, ref, 'none')
    self.assertEqual(html.strip(), 'thank you')

  def testWerIgnoreCommentPunc(self):
    ref = '(Hello world)!    [pause] Today is a good day! How are you?'
    hyp = 'hello  world. today is a good day, how are you'
    wer_obj = simple_wer.SimpleWER()
    wer_obj.AddHypRef(hyp, ref)
    err_info = wer_obj.wer_info
    self.assertEqual(err_info['del'], 0)
    self.assertEqual(err_info['sub'], 0)
    self.assertEqual(err_info['ins'], 0)
    self.assertEqual(err_info['nw'], 10)

  def testKeyPhraseCounts(self):
    key_phrases = ['Google', 'Mars']
    wer_obj = simple_wer.SimpleWER(key_phrases=key_phrases)

    ref = 'Hey  Google. I have a question about Mars, can I google it? '
    hyp = 'Hey Google! I have question about Mars, can I google it? '
    wer_obj.AddHypRef(hyp, ref)
    self.assertEqual(sum(wer_obj.ref_keyphrase_counts.values()), 3)
    self.assertEqual(sum(wer_obj.matched_keyphrase_counts.values()), 3)

    ref = 'Hey Google, could you tell me a story about Mars? '
    hyp = 'Hey Google, could you tell me a story about March? '
    wer_obj.AddHypRef(hyp, ref)
    self.assertEqual(sum(wer_obj.ref_keyphrase_counts.values()), 5)
    self.assertEqual(sum(wer_obj.matched_keyphrase_counts.values()), 4)

  def testKeyPhraseJaccardSimilarity(self):
    key_phrases = ['Google', 'Mars and Earth']
    wer_obj = simple_wer.SimpleWER(key_phrases=key_phrases)
    hyp = 'Hey Google, could you tell me a story about March and Earth?'
    ref = 'Hey Google, could you tell me a story about Mars and Earth?'
    wer_obj.AddHypRef(hyp, ref)
    stats = wer_obj.GetKeyPhraseStats()
    jaccard = stats[0]
    self.assertEqual(jaccard, 0.5)

  def testKeyPhraseF1(self):
    key_phrases = ['Google', 'Mars and Earth']
    wer_obj = simple_wer.SimpleWER(key_phrases=key_phrases)
    hyp = 'Hey Google, could you tell me a story about March and Earth?'
    ref = 'Hey Google, could you tell me a story about Mars and Earth?'
    wer_obj.AddHypRef(hyp, ref)
    stats = wer_obj.GetKeyPhraseStats()
    f1 = stats[1]
    self.assertAlmostEqual(f1, 0.66666666666666667, delta=0.01)


if __name__ == '__main__':
  tf.test.main()
