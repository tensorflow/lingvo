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
"""Tests for lingvo.core.datasource_tfds."""

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import datasource_tfds
from lingvo.core import py_utils
from lingvo.core import test_utils

import tensorflow_datasets as tfds


class TFDSMnistInputGenerator(base_input_generator.BaseInputGenerator):

  def LoadTFDSDataset(self, info, features_dict):
    example = py_utils.NestedMap.FromNestedDict(features_dict)
    example.num_classes = info.features['label'].num_classes
    return example


class TFDSInputTest(test_utils.TestCase):

  def testTFDSInput(self):
    ds_params = datasource_tfds.TFDSInput.Params().Set(
        dataset='mnist', split='train[:10]')
    ds = ds_params.Instantiate()
    with self.session():
      with tfds.testing.mock_data(num_examples=10):
        batch = ds.GetNext()
      for _ in range(10):
        res = self.evaluate(batch)
        self.assertAllEqual((28, 28, 1), res['image'].shape)
        self.assertLess(res['label'], 10)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)

  def testTFDSInputLoadFn(self):
    ds_params = datasource_tfds.TFDSInput.Params().Set(
        dataset='mnist', split='train[:10]', load_fn='LoadTFDSDataset')
    ds = ds_params.Instantiate()
    ds.SetInputGenerator(TFDSMnistInputGenerator.Params().Instantiate())
    with self.session():
      with tfds.testing.mock_data(num_examples=10):
        batch = ds.GetNext()
      for _ in range(10):
        res = self.evaluate(batch)
        self.assertAllEqual((28, 28, 1), res.image.shape)
        self.assertLess(res.label, 10)
        self.assertEqual(res.num_classes, 10)
      with self.assertRaises(tf.errors.OutOfRangeError):
        self.evaluate(batch)


if __name__ == '__main__':
  test_utils.main()
