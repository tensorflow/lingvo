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
"""Tests for tf_hub_layers.py."""

from lingvo import compat as tf
from lingvo.core import cluster_factory
from lingvo.core import test_utils
from lingvo.tasks.milan import tf_hub_layers
import tensorflow_hub as hub


def ExportFakeTF1ImageModule(*, input_image_height: int, input_image_width: int,
                             output_feature_dim: int, export_path: str):
  """Makes a TF-hub image feature module for use in unit tests.

  The resulting module has the signature of a image model, but contains a
  minimal set of trainable variables and its initialization loads nothing from
  disk.

  Args:
    input_image_height: Height of the module's input images.
    input_image_width: Width of module's input images.
    output_feature_dim: Dimension of the output feature vectors.
    export_path: Path where exported module will be written.
  """

  def ModuleFn(training):
    """Builds the graph and signature for the stub TF-hub module."""
    image_data = tf.placeholder(
        shape=[None, input_image_height, input_image_width, 3],
        dtype=tf.float32)
    # Linearly project image_data to shape [1, output_feature_dim] features.
    encoder_output = tf.compat.v1.layers.dense(
        tf.reshape(image_data,
                   [-1, input_image_height * input_image_width * 3]),
        output_feature_dim)

    # Add a non-trainable 'count' variable that can be updated through an
    # UPDATE_OP. This is analogous to a batch-norm moving average that should be
    # updated during fine-tuning.
    v = tf.get_variable('count', initializer=0, dtype=tf.int32, trainable=False)
    if training:
      update_op = v.assign_add(1).op
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    hub.add_signature(
        'default', inputs={'images': image_data}, outputs=encoder_output)

  spec = hub.create_module_spec(
      ModuleFn,
      tags_and_args=[({'train'}, dict(training=True)),
                     (set(), dict(training=False))])

  with tf.compat.v1.Graph().as_default():
    module = hub.Module(spec, trainable=True)
    with tf.compat.v1.Session() as session:
      session.run(tf.compat.v1.global_variables_initializer())
      module.export(export_path, session)


class FakeTF2ImageModule(tf.train.Checkpoint):

  def __init__(self, output_dim=768):
    # Counts the number of times the layer has been run with training=True.
    self.counter = tf.Variable(
        initial_value=0, dtype=tf.int32, name='counter', use_resource=True)
    self.output_dim = output_dim

    # "Reusable" SavedModel metadata expected by KerasLayer.
    self.variables = [self.counter]
    self.trainable_variables = []
    self.regularization_losses = []

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
      tf.TensorSpec(shape=[], dtype=tf.bool)
  ])
  def __call__(self, images, training=False):
    if training:
      self.counter.assign_add(1)
    return tf.ones([tf.shape(images)[0], self.output_dim])


class ImageModuleTest(test_utils.TestCase):

  def testImageModule(self):
    batch_size = 2
    image_size = 42
    feature_dim = 64

    export_dir = self.create_tempdir().full_path

    ExportFakeTF1ImageModule(
        input_image_height=image_size,
        input_image_width=image_size,
        output_feature_dim=feature_dim,
        export_path=export_dir)

    params = tf_hub_layers.ImageModule.Params().Set(
        name='image_module', module_path=export_dir)
    layer = params.Instantiate()
    images = tf.zeros([batch_size, image_size, image_size, 3], dtype=tf.float32)
    training_mode_features = layer(images)
    with cluster_factory.SetEval(True):
      eval_mode_features = layer(images)

    self.assertAllEqual([batch_size, feature_dim], training_mode_features.shape)
    self.assertAllEqual([batch_size, feature_dim], eval_mode_features.shape)

    # Check that update ops are run when the layer output is used during
    # training.
    count_variable = [v for v in layer.variables if v.name == 'count:0'][0]
    count_value = count_variable.read_value()

    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertEqual(0, sess.run(count_value))
      sess.run(training_mode_features)
      self.assertEqual(1, sess.run(count_value))

      sess.run(eval_mode_features)
      self.assertEqual(1, sess.run(count_value))


class ImageModuleV2Test(test_utils.TestCase):

  def testImageModuleV2(self):

    # Create a fake image encoder module in lieu of having the test download a
    # real one from tf-hub.
    export_dir = self.create_tempdir().full_path

    with self.session() as sess:
      encoder = FakeTF2ImageModule(output_dim=42)
      sess.run(tf.global_variables_initializer())
      tf.saved_model.save(encoder, export_dir)

    params = tf_hub_layers.ImageModuleV2.Params().Set(
        name='foo', module_path=export_dir)
    layer = params.Instantiate()
    images = tf.ones([2, 24, 24, 3])
    features = layer(images)
    self.assertEqual([2, 42], features.shape.as_list())

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      # Verify that calling the layer in train mode (lingvo's default) causes
      # the module's update ops to run.
      counter = layer.theta['foo/counter']
      self.assertEqual(0, counter.eval())
      _ = sess.run(features)
      _ = sess.run(features)
      self.assertEqual(2, counter.eval())

      # In eval mode, the layer should call the underlying module with
      # `training=False` and thus not run the update ops.
      with cluster_factory.SetEval(True):
        features = layer(images)
        _ = sess.run(features)
        _ = sess.run(features)
      self.assertEqual(2, counter.eval())


if __name__ == '__main__':
  tf.test.main()
