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
"""Tests for wpm_encoder."""

import os
import lingvo.compat as tf
from lingvo.core import test_utils
from lingvo.core import wpm_encoder


class WpmEncoderTest(test_utils.TestCase):

  def _CreateVocab(self):
    outpath = os.path.join(tf.test.get_temp_dir(), 'wpm.voc')
    with tf.io.gfile.GFile(outpath, 'w') as f:
      contents = [
          '<unk>',
          '<s>',
          '</s>',
          't',
          'i',
          'o',
          'f',
          'r',
          'D',
          'it',
          'or',
          'for',
          'itt',
          'to',
          'i-',
          'tt',
          'f.',
          'o-',
          'o.',
          'fo',
          'ø',  # \xC3\xB8
          'ö',  # \xC3\xB6
          '\\',
          '▁',
      ]
      f.write('\n'.join(contents))
    return outpath

  def setUp(self):
    voc = self._CreateVocab()
    self._enc = wpm_encoder.WpmEncoder(voc)

  def testDitto(self):
    with tf.Session():
      ids, strs = self._enc.Encode('Ditto')
      self.assertEqual(u'▁ D itt o'.encode('utf-8'),
                       tf.strings.reduce_join(strs, separator=' ').eval())
      self.assertEqual(b'Ditto', self._enc.Decode(ids).eval())
      ids, strs = self._enc.Encode('Ditto Ditto')
      self.assertEqual(u'▁ D itt o ▁ D itt o'.encode('utf-8'),
                       tf.strings.reduce_join(strs, separator=' ').eval())
      self.assertEqual(b'Ditto Ditto', self._enc.Decode(ids).eval())

  def testMergeProb(self):
    voc = self._CreateVocab()
    enc = wpm_encoder.WpmEncoder(voc, merge_prob=0.)
    with tf.Session():
      ids, strs = enc.Encode('Ditto')
      self.assertEqual(u'▁ D i t t o'.encode('utf-8'),
                       tf.strings.reduce_join(strs, separator=' ').eval())
      self.assertEqual(b'Ditto', self._enc.Decode(ids).eval())

  def testEmpty(self):
    with tf.Session():
      ids, strs = self._enc.Encode('')
      self.assertEqual(b'', tf.strings.reduce_join(strs, separator=' ').eval())
      self.assertEqual(b'', self._enc.Decode(ids).eval())

  def testWithBackslash(self):
    with tf.Session():
      ids, strs = self._enc.Encode('\\')
      self.assertEqual(u'▁ \\'.encode('utf-8'),
                       tf.strings.reduce_join(strs, separator=' ').eval())
      self.assertEqual(b'\\', self._enc.Decode(ids).eval())

  def testWithUnicode(self):
    with tf.Session():
      ids, strs = self._enc.Encode('føö')
      self.assertEqual(u'▁ f ø ö'.encode('utf-8'),
                       tf.strings.reduce_join(strs, separator=' ').eval())
      self.assertEqual(u'føö'.encode('utf-8'), self._enc.Decode(ids).eval())


if __name__ == '__main__':
  tf.test.main()
