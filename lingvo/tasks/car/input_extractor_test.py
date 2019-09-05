# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for input_extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import hyperparams
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.car import input_extractor
from lingvo.tasks.car import input_preprocessors


class InputExtractorTest(test_utils.TestCase):

  def testBaseExtractorRaisesErrorWithMissingPreprocessorKeys(self):
    extractors = py_utils.NestedMap()
    preprocessors = hyperparams.Params()
    preprocessors.Define(
        'count_points',
        input_preprocessors.CountNumberOfPointsInBoxes3D.Params(), '')
    preprocessors.Define('viz_copy',
                         input_preprocessors.CreateDecoderCopy.Params(), '')
    p = input_extractor.BaseExtractor.Params(extractors).Set(
        preprocessors=preprocessors,
        preprocessors_order=['count_points', 'missing_key', 'viz_copy'])
    with self.assertRaisesRegexp(
        ValueError,
        r'preprocessor_order specifies keys which were not found .*'):
      p.Instantiate()


if __name__ == '__main__':
  tf.test.main()
