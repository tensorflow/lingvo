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
"""Tests for lr_util."""

from lingvo import compat as tf
from lingvo.core import py_utils
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
    with self.session():
      # Linear ramp up.
      with py_utils.GlobalStepContext(8):
        self.assertLess(self.evaluate(schedule_layer.Value()), 1.)
      # Peak learning rate.
      with py_utils.GlobalStepContext(16):
        self.assertEqual(self.evaluate(schedule_layer.Value()), 1.)
      # Still at peak learning rate.
      with py_utils.GlobalStepContext(24):
        self.assertEqual(self.evaluate(schedule_layer.Value()), 1.)
      # Exponential ramp down.
      with py_utils.GlobalStepContext(48):
        self.assertLess(self.evaluate(schedule_layer.Value()), 1.)

  def testExponentialWithoutLinearRamp(self):
    p = self._testParams()
    lr_util.SetExponentialLR(
        p.train, p.input, exp_start_epoch=0, total_epoch=10)
    schedule_layer = p.train.lr_schedule.Instantiate()
    with self.session():
      # Peak learning rate at 0.
      with py_utils.GlobalStepContext(0):
        self.assertEqual(self.evaluate(schedule_layer.Value()), 1.)
      # Exponential ramp down within first epoch.
      with py_utils.GlobalStepContext(4):
        self.assertLess(self.evaluate(schedule_layer.Value()), 1.)

  def testCosineWithLinearRamp(self):
    p = self._testParams()
    lr_util.SetCosineLR(
        p.train, p.input, warmup_epoch=1, total_epoch=10, warmup_init=0.)
    schedule_layer = p.train.lr_schedule.Instantiate()
    with self.session():
      # Linear ramp up.
      with py_utils.GlobalStepContext(8):
        self.assertLess(self.evaluate(schedule_layer.Value()), 1.)
      # Cosine ramp down.
      with py_utils.GlobalStepContext(48):
        self.assertLess(self.evaluate(schedule_layer.Value()), 1.)


if __name__ == '__main__':
  tf.test.main()
