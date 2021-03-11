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
"""Tests for dual_encoder."""

from lingvo import compat as tf
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import test_utils
from lingvo.tasks.milan import dual_encoder
from lingvo.tasks.milan import labels as label_lib
import numpy as np


class DualEncoderTest(test_utils.TestCase):

  def _DualEncoderParamsForTest(self):
    p = dual_encoder.DualEncoder.Params()
    p.name = 'test_dual_encoder'

    x_encoder_config = dual_encoder.EncoderConfig().Set(
        input_features='x_input',
        id_feature='x_ids',
        encoder=layers.IdentityLayer.Params(),
        output_dim=3)
    y_encoder_config = dual_encoder.EncoderConfig().Set(
        input_features='y_input',
        id_feature='y_ids',
        encoder=layers.IdentityLayer.Params(),
        output_dim=4)
    p.encoder_configs = {'x': x_encoder_config, 'y': y_encoder_config}
    return p

  def testSimple(self):
    p = self._DualEncoderParamsForTest()
    p.loss_weights = {('x', 'y'): 0.5, ('y', 'x'): 0.5}
    p.joint_embedding_dim = 7

    batch_size = 2
    p.label_fn = lambda _: tf.eye(batch_size)
    model = p.Instantiate()

    input_batch = py_utils.NestedMap(
        x_input=tf.ones([batch_size, 3], dtype=tf.float32),
        x_ids=tf.range(batch_size, dtype=tf.int64),
        y_input=tf.ones([batch_size, 4], dtype=tf.float32),
        y_ids=tf.range(batch_size, dtype=tf.int64))
    preds = model.ComputePredictions(model.theta, input_batch)

    self.assertEqual([batch_size, p.joint_embedding_dim],
                     preds.x.encodings.shape.as_list())
    self.assertEqual([batch_size, p.joint_embedding_dim],
                     preds.y.encodings.shape.as_list())
    self.assertEqual(input_batch.x_ids.shape, preds.x.ids.shape)
    self.assertEqual(input_batch.y_ids.shape, preds.y.ids.shape)

  def testMaskedLoss(self):
    p = self._DualEncoderParamsForTest()
    p.encoder_configs['x'].output_dim = 5
    p.encoder_configs['y'].output_dim = 5
    x2y_weight = 0.75
    y2x_weight = 0.25
    p.loss_weights = {('x', 'y'): x2y_weight, ('y', 'x'): y2x_weight}

    # Mock the label_fn so it gives the in-batch pairs the following labels.
    x = label_lib.IGNORE_PAIR_LABEL
    example_pair_labels = tf.constant(
        [
            # pyformat: disable
            [1, 0, 0],
            [0, 1, x],
            [0, x, 1]
            # pyformat: disable
        ],
        dtype=tf.int32)
    p.label_fn = lambda _: example_pair_labels

    x_input = np.arange(15, dtype=np.float32).reshape((3, 5)) / 10.0
    y_input = np.arange(5, 20, dtype=np.float32).reshape((3, 5)) / 10.0
    model = p.Instantiate()
    input_batch = py_utils.NestedMap(
        x_input=tf.convert_to_tensor(x_input),
        x_ids=tf.constant([1, 2, 3], dtype=tf.int64),
        y_input=tf.convert_to_tensor(y_input),
        y_ids=tf.constant([4, 2, 2], dtype=tf.int64))
    # Check that pairs labeled "ignore" are excluded from the loss.
    # TODO(austinwaters): Instead of computing and checking the real loss
    # values, this test should just check that DualEncoder forwards the correct
    # inputs and labels to the loss function and applies the correct weight
    # to the result.
    expected_x2y_losses = np.array([1.6802696, 0.1602242, 0.00247565])
    expected_y2x_losses = np.array([3.6856253, 0.048587330, 0.00020346955])
    expected_average_loss = np.mean(x2y_weight * expected_x2y_losses +
                                    y2x_weight * expected_y2x_losses)

    with py_utils.GlobalStepContext(2):
      preds = model.ComputePredictions(model.theta, input_batch)
      metrics, _ = model.ComputeLoss(model.theta, preds, input_batch)
    loss, _ = metrics['loss']

    with self.session() as sess:
      loss = sess.run(loss)
    self.assertAllClose(expected_average_loss, loss)


if __name__ == '__main__':
  tf.test.main()
