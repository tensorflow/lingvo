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
"""Tools for image preprocessing.

Based on implementations in TensorFlow cloud TPU models repo.
"""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils


def _DistortBrightnessAndColor(image):
  """Distorts brightness and color of the input image.

  Args:
    image: 3-D Tensor containing single image in [0, 1].

  Returns:
    3-D Tensor color-distorted image in range [0, 1]
  """
  br_delta = tf.random.uniform([], -32. / 255., 32. / 255.)
  cb_factor = tf.random.uniform([], -0.1, 0.1)
  cr_factor = tf.random.uniform([], -0.1, 0.1)

  channels = tf.split(axis=2, num_or_size_splits=3, value=image)
  red_offset = 1.402 * cr_factor + br_delta
  green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
  blue_offset = 1.772 * cb_factor + br_delta
  channels[0] += red_offset
  channels[1] += green_offset
  channels[2] += blue_offset
  return tf.clip_by_value(tf.concat(channels, axis=2), 0., 1.)


class ImagePreprocessor(base_layer.BaseLayer):
  """Performs inception-style preprocessing on images.

  In general, inputs are assumed to be JPEG-encoded images, and outputs are RGB
  images resized to the specified `output_image_size` with values in
  `output_range`.

  The intermediate transformations differ in training and evaluation mode.
    - In training mode (the default), the images undergo random transformations.
      Namely, they are randomly cropped, color-distorted, and left-right
      reflected. The crop is chosen so that it contains at least
      `training_crop_min_area` of the original image, and is bilinearly
      resized to `output_image_size`.
    - In evaluation mode, preprocessing is deterministic. A central crop of
      `eval_crop_area` is taken and bilinearly resized to `output_image_size`.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'inception_preprocessor'
    p.Define('output_image_size', [224, 224],
             '(height, width) of output images.')
    p.Define(
        'training_crop_min_area', 0.67,
        'Minimum image area that must be in each training crop. Should be a '
        'fraction in [0, 1].')
    p.Define(
        'eval_crop_area', 0.875,
        'Area of the central crop to take when processing images for '
        'evaluation. Should be a fraction in (0, 1].')
    p.Define('parallelism', 16,
             'Number of images processed in parallel by FProp().')
    p.Define('output_range', [0, 1], 'Output range of pixel values.')
    return p

  def _PreprocessForTraining(self, image):
    """Distort one image for training a network.

    Args:
      image: The input image, a shape [height, width, num_channels=3] Tensor.
        Must be of type `tf.float32`. Image values are assumed to be in [0, 1].

    Returns:
      3-D float Tensor of distorted image used for training with range [0, 1].
    """
    p = self.params
    assert image.dtype == tf.float32

    crop_bbox_begin, crop_bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        # No objects of interest; use the whole image as input.
        bounding_boxes=tf.zeros([1, 1, 4], dtype=tf.float32),
        area_range=(p.training_crop_min_area, 1.0),
        use_image_if_no_bounding_boxes=True)
    image = tf.slice(image, crop_bbox_begin, crop_bbox_size)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    image.set_shape([None, None, 3])

    # Bilinear resize to the target shape. Note this does not respect the
    # original aspect ratio and may distort the image.
    height, width = p.output_image_size
    image = tf.image.resize(image, [height, width], antialias=True)
    image.set_shape([height, width, 3])

    image = tf.image.random_flip_left_right(image)
    image = _DistortBrightnessAndColor(image)

    # [0, 1] => output_range
    image *= float(p.output_range[1] - p.output_range[0])
    image += p.output_range[0]
    return image

  def _PreprocessForEval(self, image):
    p = self.params

    if p.eval_crop_area:
      image = tf.image.central_crop(image, central_fraction=p.eval_crop_area)

    height, width = p.output_image_size
    image = tf.image.resize(image, [height, width], antialias=True)
    image.set_shape([height, width, 3])

    image *= float(p.output_range[1] - p.output_range[0])
    image -= p.output_range[0]
    return image

  def FProp(self, _, encoded_images):
    """Decodes and preprocesses the given images.

    Args:
      encoded_images: Encoded jpeg images as a [batch_size, ...] string Tensor.

    Returns:
      The decoded images as a float32 Tensor with shape
      [batch_size, ..., height, width, num_channels=3].
    """
    p = self.params

    def _DecodeAndPreprocessOne(encoded_image):
      image = tf.image.decode_jpeg(encoded_image, channels=3)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      if self.do_eval:
        return self._PreprocessForEval(image)
      else:
        return self._PreprocessForTraining(image)

    input_shape = py_utils.GetShape(encoded_images)
    encoded_images = tf.reshape(encoded_images, [-1])
    images = tf.map_fn(
        _DecodeAndPreprocessOne,
        encoded_images,
        back_prop=False,
        dtype=tf.float32,
        parallel_iterations=p.parallelism)
    height, width = p.output_image_size
    return tf.reshape(images, input_shape + [height, width, 3])
