# Lint as: python3
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
"""Tests for car.graddrop."""

from lingvo import compat as tf
from lingvo.core import builder_layers
from lingvo.core import graddrop
from lingvo.core import test_utils


class GraddropTest(test_utils.TestCase):

  def _testGradDrop(self, graddrop_params):
    batch_size, dims = 4, 5
    gd_layer = graddrop_params.Set(name='test_gd_layer').Instantiate()
    linear_layer = builder_layers.LinearLayer.Params().Set(
        name='test_linear_layer', input_dims=dims,
        output_dims=dims).Instantiate()

    x = tf.random.uniform((batch_size, dims))
    x = linear_layer.FPropDefaultTheta(x)

    # Make a copy of x after graddrop.
    x_gd = gd_layer.FPropDefaultTheta(x)

    # Compute a loss based on graddrop's version of x.
    gd_loss_0 = tf.reduce_sum(x_gd**2)
    gd_loss_1 = tf.reduce_sum(-tf.abs(x_gd))
    gd_layer.SetLosses([
        (gd_loss_0, 0.1),
        (gd_loss_1, 0.2),
    ])
    gd_total_loss = gd_loss_0 + gd_loss_1
    gd_grad = tf.gradients(gd_total_loss, x)

    # Compute the same loss based on the regular version of x.
    loss_0 = tf.reduce_sum(x**2)
    loss_1 = tf.reduce_sum(-tf.abs(x))
    total_loss = loss_0 + loss_1
    grad = tf.gradients(total_loss, x)

    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      actual_total_loss, actual_grad, actual_gd_total_loss, actual_gd_grad = (
          sess.run([total_loss, grad, gd_total_loss, gd_grad]))

    # Verify that losses are similar, but the gradients are different.
    self.assertAllClose(actual_total_loss, actual_gd_total_loss)
    self.assertNotAllClose(actual_grad, actual_gd_grad)

  def testGradDropBasic(self):
    self._testGradDrop(graddrop.GradDrop.Params())

  def testGradDropSigmoidKeepProb(self):
    self._testGradDrop(
        graddrop.GradDrop.Params().Set(keep_prob_function='sigmoid'))

  def testGradDropInvalidKeepProb(self):
    with self.assertRaisesRegex(ValueError, r'.*keep_prob_function must be.*'):
      self._testGradDrop(
          graddrop.GradDrop.Params().Set(keep_prob_function='invalid'))

  def testGradDropUseInputScale(self):
    self._testGradDrop(
        graddrop.GradDrop.Params().Set(use_input_sign_only=False))

  def testGradDropAllowNormChange(self):
    self._testGradDrop(
        graddrop.GradDrop.Params().Set(keep_gradnorm_constant=False))

  def testGradDropSkipMarginalizeBatchDim(self):
    self._testGradDrop(
        graddrop.GradDrop.Params().Set(marginalize_batch_dim=False))

  def testGradDropFPropTwiceRaisesError(self):
    batch_size, dims = 4, 5
    gd_layer = graddrop.GradDrop.Params().Set(
        name='test_gd_layer').Instantiate()
    x = tf.random.uniform((batch_size, dims))
    gd_layer.FPropDefaultTheta(x)
    with self.assertRaisesRegex(ValueError, r'.*FProp was already called.*'):
      gd_layer.FPropDefaultTheta(x)

  def testGradDropSetLossesTwiceRaisesError(self):
    batch_size, dims = 4, 5
    gd_layer = graddrop.GradDrop.Params().Set(
        name='test_gd_layer').Instantiate()
    x = tf.random.uniform((batch_size, dims))
    x_gd = gd_layer.FPropDefaultTheta(x)
    gd_loss_0 = tf.reduce_sum(x_gd**2)
    gd_loss_1 = tf.reduce_sum(-tf.abs(x_gd))
    gd_layer.SetLosses([
        (gd_loss_0, 0.1),
        (gd_loss_1, 0.2),
    ])
    with self.assertRaisesRegex(ValueError, r'.*Losses already set.*'):
      gd_layer.SetLosses([
          (gd_loss_0, 0.1),
          (gd_loss_1, 0.2),
      ])


if __name__ == '__main__':
  tf.test.main()
