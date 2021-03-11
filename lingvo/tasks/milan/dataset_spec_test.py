# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for dataset_spec.py."""

from lingvo import compat as tf
from lingvo.core import test_utils

from lingvo.tasks.milan import dataset_spec


class TFRecordDatasetSpecTest(test_utils.TestCase):

  def testPopulatesFeaturesInMetadata(self):
    spec = dataset_spec.TFRecordDatasetSpec(
        # Create a file so Dataset.list_files() won't complain if the test is
        # ever switched to eager.
        split_paths={'train': self.create_tempfile().full_path},
        schema={
            'quizybuck':
                tf.io.FixedLenFeature([42], tf.int64),
            'x':
                tf.io.FixedLenSequenceFeature([3],
                                              tf.float32,
                                              allow_missing=True)
        },
        label_fn=None)
    expected_features = {
        'quizybuck': tf.TensorSpec([42], tf.int64),
        'x': tf.TensorSpec([None, 3], tf.float32)
    }
    self.assertSameStructure(spec.meta.features, expected_features)

    # Check that these match the features in the actual `tf.data.Dataset`.
    self.assertSameStructure(spec.Read('train').element_spec, expected_features)


if __name__ == '__main__':
  tf.test.main()
