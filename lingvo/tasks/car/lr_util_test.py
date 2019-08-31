# Lint as: python2, python3
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
"""Tests for lr_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo import compat as tf
from lingvo.core import test_utils
from lingvo.tasks.car import kitti_input_generator
from lingvo.tasks.car import lr_util
from lingvo.tasks.car import pillars


class LrUtilTest(test_utils.TestCase):

  def _testInput(self):
    p = kitti_input_generator.KITTIGrid.Params()
    p.batch_size = 8
    p.num_samples = 128
    return p

  def _testParams(self):
    p = pillars.ModelV1.Params()
    p.name = 'test'
    p.input = self._testInput()
    return p

  def testExponentialWithLinearRamp(self):
    p = self._testParams()
    lr_util.SetExponentialLR(
        p.train,
        p.input,
        warmup_epoch=1,
        exp_start_epoch=2,
        total_epoch=10,
        warmup_init=0.)
    schedule_layer = p.train.lr_schedule.Instantiate()
    with self.session() as sess:
      # Linear ramp up.
      self.assertLess(sess.run(schedule_layer.Value(8)), 1.)
      # Peak learning rate.
      self.assertEqual(sess.run(schedule_layer.Value(16)), 1.)
      # Still at peak learning rate.
      self.assertEqual(sess.run(schedule_layer.Value(24)), 1.)
      # Exponential ramp down.
      self.assertLess(sess.run(schedule_layer.Value(48)), 1.)

  def testExponentialWithoutLinearRamp(self):
    p = self._testParams()
    lr_util.SetExponentialLR(
        p.train, p.input, exp_start_epoch=0, total_epoch=10)
    schedule_layer = p.train.lr_schedule.Instantiate()
    with self.session() as sess:
      # Peak learning rate at 0.
      self.assertEqual(sess.run(schedule_layer.Value(0)), 1.)
      # Exponential ramp down within first epoch.
      self.assertLess(sess.run(schedule_layer.Value(4)), 1.)


if __name__ == '__main__':
  tf.test.main()
