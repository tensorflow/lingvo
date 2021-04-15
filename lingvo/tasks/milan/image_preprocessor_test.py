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
"""Tests for image_preprocessor.py."""

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import cluster_factory
from lingvo.core import test_utils

from lingvo.tasks.milan import image_preprocessor


def _EncodeRandomJpegs(sizes):
  images = [
      tf.cast(
          tf.random.uniform([height, width, 3], maxval=256, dtype=tf.int32),
          tf.uint8) for height, width in sizes
  ]
  return tf.stack([tf.io.encode_jpeg(image) for image in images])


class ImagePreprocessorTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Eval', True), ('Training', False))
  def testPreprocessor(self, use_eval_mode):
    params = image_preprocessor.ImagePreprocessor.Params()

    encoded_images = _EncodeRandomJpegs(sizes=[(24, 42), (17, 19)])
    preprocessor = params.Instantiate()
    with cluster_factory.SetEval(use_eval_mode):
      images = preprocessor(encoded_images)
      self.assertEqual(use_eval_mode, preprocessor.do_eval)
      self.assertAllEqual(
          list(encoded_images.shape) + params.output_image_size + [3],
          images.shape)

  def testEncodedSingleImageShape(self):
    params = image_preprocessor.ImagePreprocessor.Params()

    encoded_images = _EncodeRandomJpegs(sizes=[(24, 42)])
    preprocessor = params.Instantiate()
    encoded_images = tf.reshape(encoded_images, [])
    images = preprocessor(encoded_images)
    self.assertAllEqual(encoded_images.shape + params.output_image_size + [3],
                        images.shape)

  def testEncodedMultipleImagesShape(self):
    params = image_preprocessor.ImagePreprocessor.Params()

    encoded_images = _EncodeRandomJpegs(sizes=[(24, 42), (17, 19), (24,
                                                                    42), (17,
                                                                          19)])
    encoded_images = tf.reshape(encoded_images, [2, 2])
    preprocessor = params.Instantiate()
    images = preprocessor(encoded_images)
    self.assertAllEqual(encoded_images.shape + params.output_image_size + [3],
                        images.shape)

  @parameterized.parameters(([0, 1],), ([-1, 1],), ([4.25, 7.75],))
  def testPreprocessorOutputRange(self, output_range):
    """Verifies that outputs image values are in `output_range`."""
    params = image_preprocessor.ImagePreprocessor.Params().Set(
        output_range=output_range)
    preprocessor = params.Instantiate()
    images = preprocessor(_EncodeRandomJpegs(sizes=[(24, 42), (17, 19)]))
    self.assertAllInRange(images, params.output_range[0],
                          params.output_range[1])


if __name__ == '__main__':
  tf.test.main()
