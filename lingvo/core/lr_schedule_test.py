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
"""Tests for lr_schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from six.moves import range

import tensorflow as tf
from lingvo.core import cluster_factory
from lingvo.core import early_stop
from lingvo.core import lr_schedule
from lingvo.core import test_utils


class LearningRateScheduleTest(test_utils.TestCase):

  def testConstantOne(self):
    with self.session(use_gpu=False):
      p = lr_schedule.ConstantOne.Params()
      lrs = p.cls(p)
      for x in [0, 10, 100, 1000000]:
        self.assertAllClose(lrs.Value(x).eval(), 1.0)

  def testPiecewiseConstant(self):
    cls = lr_schedule.PiecewiseConstantLearningRateSchedule
    with self.session(use_gpu=False):
      bs = [300000, 400000, 500000]
      vs = [1.0, 0.1, 0.01, 0.001]
      x_ins = [tf.constant(x) for x in [299999, 399999, 499999, 599999]]
      outs = []
      for x in x_ins:
        lrs = cls(cls.Params().Set(boundaries=bs, values=vs))
        outs.append(lrs.Value(x).eval())
      self.assertAllClose([1.0, 0.1, 0.01, 0.001], outs)

  def testContinuousLearningRateSchedule(self):
    p = lr_schedule.ContinuousLearningRateSchedule.Params()
    p.start_step = 1000
    p.half_life_steps = 100
    p.min = 0.1
    decay = p.cls(p)
    with self.session():
      self.assertAllClose(decay.Value(0).eval(), 1.0)
      self.assertAllClose(decay.Value(500).eval(), 1.0)
      self.assertAllClose(decay.Value(1000).eval(), 1.0)
      self.assertAllClose(decay.Value(1100).eval(), 0.5)
      self.assertAllClose(decay.Value(1200).eval(), 0.25)
      self.assertAllClose(decay.Value(1300).eval(), 0.125)
      self.assertAllClose(decay.Value(1400).eval(), 0.1)
      self.assertAllClose(decay.Value(2000).eval(), 0.1)

      # Tests that the decay consistently decreases by half per 100
      # steps.
      for step in range(1000, 1200, 25):
        self.assertGreater(
            decay.Value(step).eval(),
            decay.Value(step + 10).eval())
        self.assertAllClose(
            decay.Value(step).eval(),
            decay.Value(step + 100).eval() * 2.)

  def testContinuousLearningRateSchedule_CanOverrideStart(self):
    p = lr_schedule.ContinuousLearningRateSchedule.Params()
    p.initial_value = 2.0
    p.start_step = 1000
    p.half_life_steps = 100
    decay = p.cls(p)
    with self.session():
      self.assertAllClose(decay.Value(0).eval(), 2.0)
      self.assertAllClose(decay.Value(1000).eval(), 2.0)
      self.assertAllClose(decay.Value(1100).eval(), 1.0)
      self.assertAllClose(decay.Value(1200).eval(), 0.5)
      self.assertAllClose(decay.Value(1300).eval(), 0.25)

  def testStepwiseExponentialSchedule(self):
    p = lr_schedule.StepwiseExponentialSchedule.Params()
    p.decay = 0.5
    p.num_steps_per_decay = 1000
    decay = p.cls(p)
    with self.session():
      self.assertAllClose(decay.Value(0).eval(), 1.0)
      self.assertAllClose(decay.Value(999).eval(), 1.0)
      self.assertAllClose(decay.Value(1000).eval(), 0.5)
      self.assertAllClose(decay.Value(1999).eval(), 0.5)
      self.assertAllClose(decay.Value(2000).eval(), 0.25)

  def testTransformerLearningRateSchedule(self):
    p = lr_schedule.TransformerLearningRateSchedule.Params()
    p.warmup_steps = 4000
    p.model_dim = 512
    lrs = p.cls(p)
    with self.session():
      print(lrs.Value(0).eval())
      print(lrs.Value(1000).eval())
      print(lrs.Value(2000).eval())
      print(lrs.Value(3000).eval())
      print(lrs.Value(4000).eval())
      print(lrs.Value(4500).eval())
      print(lrs.Value(5000).eval())
      self.assertAllClose(lrs.Value(0).eval(), 1.74693e-07)
      self.assertAllClose(lrs.Value(1000).eval(), 0.000174867)
      self.assertAllClose(lrs.Value(2000).eval(), 0.00034956)
      self.assertAllClose(lrs.Value(3000).eval(), 0.000524253)
      self.assertAllClose(lrs.Value(4000).eval(), 0.000698684)
      self.assertAllClose(lrs.Value(4500).eval(), 0.000658735)
      self.assertAllClose(lrs.Value(5000).eval(), 0.000624937)
      # Tests that the schedule peaks at 4000 steps.
      self.assertGreater(lrs.Value(4000).eval(), lrs.Value(3990).eval())
      self.assertGreater(lrs.Value(4000).eval(), lrs.Value(4010).eval())

      # Tests that the schedule increases linearly before 4000 steps.
      for step in range(300, 4000, 200):
        self.assertAllClose(
            lrs.Value(step).eval() * 2.,
            lrs.Value(step + 10).eval() + lrs.Value(step - 10).eval())

  def testTransformerLearningRateScheduleWithDecayEnd(self):
    p = lr_schedule.TransformerLearningRateSchedule.Params()
    p.warmup_steps = 4000
    p.model_dim = 512
    p.decay_end = 5000
    lrs = p.cls(p)
    with self.session():
      self.assertAllClose(lrs.Value(0).eval(), 1.74693e-07)
      self.assertAllClose(lrs.Value(3000).eval(), 0.000524253)
      self.assertAllClose(lrs.Value(5000).eval(), 0.000624937)

      # Tests that the schedule peaks at 4000 steps.
      self.assertGreater(lrs.Value(4000).eval(), lrs.Value(3990).eval())
      self.assertGreater(lrs.Value(4000).eval(), lrs.Value(4010).eval())

      # Tests that the schedule increases linearly before 4000 steps.
      for step in range(300, 4000, 200):
        self.assertAllClose(
            lrs.Value(step).eval() * 2.,
            lrs.Value(step + 10).eval() + lrs.Value(step - 10).eval())

      print(lrs.Value(4999).eval())
      print(lrs.Value(5000).eval())
      print(lrs.Value(5001).eval())
      print(lrs.Value(6000).eval())
      # Tests that the schedule is fixed after decay end steps.
      self.assertGreater(lrs.Value(4999).eval(), lrs.Value(5000).eval())
      self.assertAllClose(lrs.Value(5000).eval(), lrs.Value(5001).eval())
      self.assertAllClose(lrs.Value(5000).eval(), lrs.Value(6000).eval())

  def testTransformerLearningRateScheduleNoWarmUp(self):
    params = lr_schedule.TransformerLearningRateScheduleNoWarmUp.Params().Set(
        decay_start=4000, model_dim=512)
    lrs = params.cls(params)

    base_params = lr_schedule.TransformerLearningRateSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    base_lrs = base_params.cls(base_params)

    with self.session():

      # Tests that the schedule is flat up until 4000 steps.
      self.assertAllClose(lrs.Value(0).eval(), 0.000698684)
      self.assertAllClose(lrs.Value(1000).eval(), 0.000698684)
      self.assertAllClose(lrs.Value(2000).eval(), 0.000698684)
      self.assertAllClose(lrs.Value(3000).eval(), 0.000698684)
      self.assertAllClose(lrs.Value(4000).eval(), 0.000698684)
      self.assertAllClose(lrs.Value(4500).eval(), 0.000658735)
      self.assertAllClose(lrs.Value(5000).eval(), 0.000624937)

      # Test that the schedule is identical with transformer-lr after 4k steps
      self.assertAllClose(base_lrs.Value(4000).eval(), lrs.Value(4000).eval())
      self.assertAllClose(base_lrs.Value(4010).eval(), lrs.Value(4010).eval())
      self.assertAllClose(base_lrs.Value(5000).eval(), lrs.Value(5000).eval())

  def testPolynomialLRSchedule(self):
    p = lr_schedule.PolynomialLearningRateSchedule.Params().Set(
        power=2, start=(0, 0.), limit=(20000, 2.))
    with self.session():
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in [0, 10000, 20000]]
      self.assertAllClose(
          pts,
          [
              [0, 0.0],
              [10000, 0.5],  # 2 * (0.5 ** 2)
              [20000, 2.0],
          ])

  def testCombinedLRSchedule(self):
    p = lr_schedule.CombinedMinimumLearningRateSchedule.Params().Set(schedules=[
        lr_schedule.LinearLearningRateSchedule.Params().Set(
            start=(0., 1.), limit=(2000000, 8.)),
        lr_schedule.LinearLearningRateSchedule.Params().Set(
            start=(2000000., 8.), limit=(4000000, 8.)),
        lr_schedule.ExponentialLearningRateSchedule.Params().Set(
            start=(4000000., 8.), limit=(8000000, 0.5))
    ])
    with self.session():
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 10000000, 1000000)]
      self.assertAllClose(
          pts,
          [
              # Linear increasing.
              [0, 1.0],
              [1000000, 4.5],
              # Constant
              [2000000, 8.0],
              [3000000, 8.0],
              # Exponentially decreasing.
              [4000000, 8.0],
              [5000000, 4.0],
              [6000000, 2.0],
              [7000000, 1.0],
              [8000000, 0.5],
              [9000000, 0.5]
          ])

  def testLinearRampupExponentialDecayScaledByNumSplitSchedule(self):
    p = lr_schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=250000, decay_start=32000000, decay_end=64000000, min=0.5)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 10000000, 1000000)]
      self.assertAllClose(
          pts,
          [
              # Linear increasing.
              [0, 1.0],
              [1000000, 4.5],
              # Constant
              [2000000, 8.0],
              [3000000, 8.0],
              # Exponentially decreasing.
              [4000000, 8.0],
              [5000000, 4.0],
              [6000000, 2.0],
              [7000000, 1.0],
              [8000000, 0.5],
              [9000000, 0.5]
          ])

  def testLinearRampupExponentialDecayScaledByNumSplitScheduleWarmUpInit(self):
    p = lr_schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup_init=0,
        warmup=250000,
        decay_start=32000000,
        decay_end=64000000,
        min=0.5)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 10000000, 1000000)]
      self.assertAllClose(
          pts,
          [
              # Linear increasing from warmup_init=0.
              [0, 0],
              [1000000, 4.0],
              # Constant
              [2000000, 8.0],
              [3000000, 8.0],
              # Exponentially decreasing.
              [4000000, 8.0],
              [5000000, 4.0],
              [6000000, 2.0],
              [7000000, 1.0],
              [8000000, 0.5],
              [9000000, 0.5]
          ])

  def testLinearRampupExponentialDecayScaledByNumSplitScheduleWithCap(self):
    p = lr_schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=250000,
        decay_start=32000000,
        decay_end=64000000,
        min=0.5,
        max=5.0)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 10000000, 1000000)]
      self.assertAllClose(
          pts,
          [
              # Linear increasing.
              [0, 1.0],
              [1000000, 4.5],
              # Constant
              [2000000, 5.0],
              [3000000, 5.0],
              # Exponentially decreasing.
              [4000000, 5.0],
              [5000000, 4.0],
              [6000000, 2.0],
              [7000000, 1.0],
              [8000000, 0.5],
              [9000000, 0.5]
          ])

  def testLinearRampupExponentialDecayScaledByNumSplitScheduleWithNumSplits(
      self):
    p = lr_schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=250000,
        decay_start=32000000,
        decay_end=64000000,
        min=0.5,
        max=5.0,
        num_splits=8)
    # Increases the number of splits to 32.
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8, split_size=4):
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 10000000, 1000000)]
      # Values are copied from
      # testLinearRampupExponentialDecayScaledByNumSplitScheduleWithCap.
      self.assertAllClose(
          pts,
          [
              # Linear increasing.
              [0, 1.0],
              [1000000, 4.5],
              # Constant
              [2000000, 5.0],
              [3000000, 5.0],
              # Exponentially decreasing.
              [4000000, 5.0],
              [5000000, 4.0],
              [6000000, 2.0],
              [7000000, 1.0],
              [8000000, 0.5],
              [9000000, 0.5]
          ])

  def testDevBasedSchedule(self):
    logdir = tf.test.get_temp_dir()
    tf.gfile.MkDir(os.path.join(logdir, 'eval_dev'))

    p = lr_schedule.DevBasedSchedule.Params()
    p.tolerance = 1.0
    p.window = 2
    p.decay = 0.5
    p.min_factor = 0.20
    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)

    lrs = p.cls(p)
    mh = lrs._metric_history
    mh.params.local_filesystem = True
    with self.session():
      tf.global_variables_initializer().run()
      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 1, 10.0)
      # best = 1
      self.assertAllClose(lrs.Value(0).eval(), 1.0)

      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 2, 5.0)
      # best = 2
      self.assertAllClose(lrs.Value(0).eval(), 1.0)

      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 5, 4.0)
      # best = 2, out of window
      self.assertAllClose(lrs.Value(0).eval(), 0.5)

      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 6, 4.0)
      # best = 2, ref = 5, in window
      self.assertAllClose(lrs.Value(0).eval(), 0.5)

      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 9, 4.0)
      # best = 2, ref = 5, out of window
      self.assertAllClose(lrs.Value(0).eval(), 0.25)

      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 10, 3.9)
      # best = 10
      self.assertAllClose(lrs.Value(0).eval(), 0.25)

      mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 13, 3.0)
      # best = 10, out of window, min factor
      self.assertAllClose(lrs.Value(0).eval(), 0.20)

  def testLinearRampupPiecewiseConstantSchedule(self):
    p = lr_schedule.LinearRampupPiecewiseConstantSchedule.Params().Set(
        boundaries=[40, 64, 80, 96],
        lrs=[1.0, 0.1, 0.01, 0.001],
    )
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', tpus=8):
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 15, 1)]

      self.assertAllClose(
          pts, [[0, 0.0], [1, 1.6], [2, 3.2], [3, 4.8], [4, 6.4], [5, 8.0],
                [6, 8.0], [7, 8.0], [8, 8.], [9, 0.8], [10, 0.8], [11, 0.08],
                [12, 0.08], [13, 0.008], [14, 0.008]])

  def testCosineSchedule(self):
    p = lr_schedule.CosineSchedule.Params().Set(
        initial_value=3.0, final_value=1.0, total_steps=400000)
    with self.session():
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 600000, 100000)]
      self.assertAllClose(
          pts,
          [
              [0, 3.0],
              [100000, math.cos(math.pi / 4) + 2.],  # angle=pi/4
              [200000, 2.0],  # angle=pi/2, half-way
              [300000, math.cos(math.pi * 3 / 4) + 2.],  # angle=pi*3/4
              [400000, 1.0],
              [500000, 1.0],
          ])

  def testLinearRampupCosineSchedule(self):
    p = lr_schedule.LinearRampupCosineSchedule.Params().Set(
        warmup_steps=200, initial_value=2.0, total_steps=400000)
    with self.session():
      lrs = p.cls(p)

      pts = [[i, lrs.Value(i).eval()]
             for i in [0, 100, 200, 100000, 200000, 300000, 400000]]
      self.assertAllClose(
          pts,
          [
              [0, 0.0],
              [100, 1.0],
              [200, 2.0],
              [100000, math.cos(math.pi / 4) + 1.],  # angle=pi/4
              [200000, 1.0],  # angle=pi/2, half-way
              [300000, math.cos(math.pi * 3 / 4) + 1.],  # angle=pi*3/4
              [400000, 0.0],
          ])

  def testPiecewiseSchedule(self):
    # Linear ramp-up in 20000 steps, cosine decay in 40000 steps.
    p0 = lr_schedule.LinearLearningRateSchedule.Params().Set(
        start=(0, 0.), limit=(20000, 2.))
    p1 = lr_schedule.CosineSchedule.Params().Set(
        initial_value=2.0, total_steps=40000)
    p = lr_schedule.PiecewiseSchedule.Params().Set(
        boundaries=[20000], schedules=[p0, p1])
    with self.session():
      lrs = p.cls(p)
      pts = [[i, lrs.Value(i).eval()] for i in range(0, 70000, 10000)]
      self.assertAllClose(
          pts,
          [
              [0, 0.0],
              [10000, 1.0],  # half-way in linear ramp-up.
              [20000, 2.0],  # completed linear ramp-up.
              [30000, math.cos(math.pi / 4) + 1.],  # pi/4.
              [40000, 1.0],  # pi/2.
              [50000, math.cos(math.pi * 3 / 4) + 1.],  # pi*3/4.
              [60000, 0.0],  # pi.
          ])


if __name__ == '__main__':
  tf.test.main()
