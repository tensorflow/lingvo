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
"""Tests for base_model_params."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lingvo.core import base_model_params
from lingvo.core import test_utils


class BaseModelParamsTest(test_utils.TestCase):

  def testGetDatasetParams_SingleTaskModelParams(self):
    dummy_model = base_model_params.SingleTaskModelParams
    self.assertEqual(dummy_model.Train(), dummy_model.GetDatasetParams('Train'))
    self.assertEqual(dummy_model.Dev(), dummy_model.GetDatasetParams('Dev'))
    self.assertEqual(dummy_model.Test(), dummy_model.GetDatasetParams('Test'))
    with self.assertRaises(AttributeError):
      dummy_model.GetDatasetParams('Invalid')

  def testGetDatasetParams_MultiTaskModelParams(self):
    dummy_model = base_model_params.MultiTaskModelParams
    self.assertEqual(dummy_model.Train(), dummy_model.GetDatasetParams('Train'))
    self.assertEqual(dummy_model.Dev(), dummy_model.GetDatasetParams('Dev'))
    self.assertEqual(dummy_model.Test(), dummy_model.GetDatasetParams('Test'))
    with self.assertRaises(AttributeError):
      dummy_model.GetDatasetParams('Invalid')


if __name__ == '__main__':
  tf.test.main()
