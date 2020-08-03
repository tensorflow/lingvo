# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for EG-DD optimizer."""
from lingvo import compat as tf
from lingvo.core import egdd
from lingvo.core import test_utils


class EGDD(test_utils.TestCase):

  def testDenseLayer(self):
    """EG-DD update."""

    with self.cached_session() as sess:
      var = tf.Variable([0.5, 1.0])
      grad = tf.placeholder(tf.float32, shape=[2])
      opt = egdd.EGDD(
          learning_rate=0.1,
          momentum=0.9,
          beta=0.1,
          gain_learning_rate=1e-2,
          scale_learning_rate=1e-3,
          use_signs=False)

      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      pre_momentum = sess.run(opt.get_slot(var, 'momentum'))
      pre_gain = sess.run(opt.get_slot(var, 'gain'))
      pre_lr_scale = sess.run(opt.get_slot(var, 'lr_scale'))
      self.assertAllClose([0.5, 1.0], pre_var)
      self.assertAllClose([0.0, 0.0], pre_momentum)
      self.assertAllClose([1.0, 1.0], pre_gain)
      self.assertAllClose([1.0], pre_lr_scale)
      sess.run(step, feed_dict={grad: [0.1, -0.5]})
      pre_var = sess.run(var)
      pre_momentum = sess.run(opt.get_slot(var, 'momentum'))
      pre_gain = sess.run(opt.get_slot(var, 'gain'))
      pre_lr_scale = sess.run(opt.get_slot(var, 'lr_scale'))
      self.assertAllClose([0.49, 1.05], pre_var)
      self.assertAllClose([0.01, -0.05], pre_momentum)
      self.assertAllClose([1, 1], pre_gain)
      self.assertAllClose([1.0], pre_lr_scale)
      sess.run(step, feed_dict={grad: [-1.0, -1.5]})
      pre_var = sess.run(var)
      pre_momentum = sess.run(opt.get_slot(var, 'momentum'))
      pre_gain = sess.run(opt.get_slot(var, 'gain'))
      pre_lr_scale = sess.run(opt.get_slot(var, 'lr_scale'))
      self.assertAllClose([0.5810, 1.2463], pre_var, atol=1e-4)
      self.assertAllClose([-0.0909, -0.1961], pre_momentum, atol=1e-4)
      self.assertAllClose([0.9990, 1.0075], pre_gain, atol=1e-4)
      self.assertAllClose([1.0007], pre_lr_scale, atol=1e-4)

  def testDenseLayerSigns(self):
    """EG-DD update."""

    with self.cached_session() as sess:
      var = tf.Variable([0.5, 1.0])
      grad = tf.placeholder(tf.float32, shape=[2])
      opt = egdd.EGDD(
          learning_rate=0.1,
          momentum=0.9,
          beta=0.1,
          gain_learning_rate=1e-2,
          scale_learning_rate=1e-3,
          use_signs=True)

      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      pre_momentum = sess.run(opt.get_slot(var, 'momentum'))
      pre_gain = sess.run(opt.get_slot(var, 'gain'))
      pre_lr_scale = sess.run(opt.get_slot(var, 'lr_scale'))
      self.assertAllClose([0.5, 1.0], pre_var)
      self.assertAllClose([0.0, 0.0], pre_momentum)
      self.assertAllClose([1.0, 1.0], pre_gain)
      self.assertAllClose([1.0], pre_lr_scale)
      sess.run(step, feed_dict={grad: [0.1, -0.5]})
      pre_var = sess.run(var)
      pre_momentum = sess.run(opt.get_slot(var, 'momentum'))
      pre_gain = sess.run(opt.get_slot(var, 'gain'))
      pre_lr_scale = sess.run(opt.get_slot(var, 'lr_scale'))
      self.assertAllClose([0.49, 1.05], pre_var)
      self.assertAllClose([0.01, -0.05], pre_momentum)
      self.assertAllClose([1, 1], pre_gain)
      self.assertAllClose([1.0], pre_lr_scale)
      sess.run(step, feed_dict={grad: [-1.0, -1.5]})
      pre_var = sess.run(var)
      pre_momentum = sess.run(opt.get_slot(var, 'momentum'))
      pre_gain = sess.run(opt.get_slot(var, 'gain'))
      pre_lr_scale = sess.run(opt.get_slot(var, 'lr_scale'))
      self.assertAllClose([0.5801, 1.2466], pre_var, atol=1e-4)
      self.assertAllClose([-0.0900, -0.1965], pre_momentum, atol=1e-4)
      self.assertAllClose([0.9900, 1.0101], pre_gain, atol=1e-4)
      self.assertAllClose([1.0007], pre_lr_scale, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
