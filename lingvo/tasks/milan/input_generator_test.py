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
"""Tests for input_generator."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import test_utils
from lingvo.tasks.milan import input_generator


class FakePreprocessor(base_layer.BaseLayer):
  """Preprocessor that fills its input feature with value 42."""

  def FProp(self, _, inputs):
    return tf.fill(tf.shape(inputs), 42)


class InputGeneratorTest(test_utils.TestCase):

  def testPreprocessing(self):

    def ReadFakeDataset(batch_size=None):
      """Simple dataset_fn for MilanInputGenerator."""
      return tf.data.Dataset.from_tensor_slices({
          'image/encoded': ['image1', 'image2', 'image3'],
          'other/feature/to/be/filtered/out': [1, 2, 3],
      }).repeat(-1).batch(batch_size)

    params = input_generator.MilanInputGenerator.Params()
    params.dataset_fn = ReadFakeDataset
    params.features_to_read = ['image/encoded']
    params.preprocessors['image/encoded'] = FakePreprocessor.Params()

    batch = params.Instantiate().GetPreprocessedInputBatch()
    self.assertAllEqual(['image/encoded'], list(batch.keys()))
    with self.session() as sess:
      feature_values = sess.run(batch['image/encoded'])
    self.assertAllEqual([42] * params.batch_size, feature_values)


if __name__ == '__main__':
  tf.test.main()
