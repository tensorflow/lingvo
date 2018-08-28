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
"""Input generator for image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tensorflow as tf

from tensorflow.python.ops import io_ops
from lingvo.core import base_input_generator
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core.ops import py_x_ops


class _MnistInputBase(base_input_generator.BaseTinyDatasetInput):
  """Base input params for MNIST."""

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(_MnistInputBase, cls).Params()
    p.data_dtype = tf.uint8
    p.data_shape = (28, 28, 1)
    p.label_dtype = tf.uint8
    return p

  def _Preprocess(self, raw):
    data = tf.stack(
        [tf.image.per_image_standardization(img) for img in tf.unstack(raw)])
    data.set_shape(raw.shape)
    return data


class MnistTrainInput(_MnistInputBase):
  """MNist training set."""

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(MnistTrainInput, cls).Params()
    p.data = 'x_train'
    p.label = 'y_train'
    p.num_samples = 60000
    p.batch_size = 256
    p.repeat = True
    return p


class MnistTestInput(_MnistInputBase):
  """MNist test set."""

  @classmethod
  def Params(cls):
    """Defaults params."""
    p = super(MnistTestInput, cls).Params()
    p.data = 'x_test'
    p.label = 'y_test'
    p.num_samples = 10000
    p.batch_size = 256
    p.repeat = False
    return p


def FakeMnistData(train_size=60000, test_size=10000):
  """Fake Mnist data for unit tests."""
  tmpdir = tempfile.mkdtemp()
  data_path = os.path.join(tmpdir, 'ckpt')
  with tf.Graph().as_default():
    with tf.Session() as sess:
      x_train = tf.ones((train_size, 28, 28, 1), dtype=tf.uint8)
      y_train = tf.ones((train_size), dtype=tf.uint8)
      x_test = tf.ones((test_size, 28, 28, 1), dtype=tf.uint8)
      y_test = tf.ones((test_size), dtype=tf.uint8)
      sess.run(
          io_ops.save_v2(data_path, ['x_train', 'y_train', 'x_test', 'y_test'],
                         [''] * 4, [x_train, y_train, x_test, y_test]))
  return tmpdir, data_path
