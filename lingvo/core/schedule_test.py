# Lint as: python3
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

import math
import os
import lingvo.compat as tf
from lingvo.core import cluster_factory
from lingvo.core import early_stop
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import test_utils


class LearningRateScheduleTest(test_utils.TestCase):

  def testConstant(self):
    with self.session(use_gpu=False):
      p = schedule.Constant.Params().Set(value=5)
      lrs = p.Instantiate()
      for x in [0, 10, 100, 1000000]:
        with py_utils.GlobalStepContext(x):
          self.assertAllClose(lrs.Value().eval(), 5.0)

  def testConstantOne(self):
    with self.session(use_gpu=False):
      p = schedule.ConstantOne.Params()
      lrs = p.Instantiate()
      for x in [0, 10, 100, 1000000]:
        with py_utils.GlobalStepContext(x):
          self.assertAllClose(lrs.Value().eval(), 1.0)

  def testPiecewiseConstant(self):
    cls = schedule.PiecewiseConstantSchedule
    with self.session(use_gpu=False):
      bs = [300000, 400000, 500000]
      vs = [1.0, 0.1, 0.01, 0.001]
      x_ins = [tf.constant(x) for x in [299999, 399999, 499999, 599999]]
      outs = []
      for x in x_ins:
        with py_utils.GlobalStepContext(x):
          lrs = cls.Params().Set(boundaries=bs, values=vs).Instantiate()
          outs.append(lrs.Value().eval())
      self.assertAllClose([1.0, 0.1, 0.01, 0.001], outs)

  def testContinuousSchedule(self):
    p = schedule.ContinuousSchedule.Params()
    p.start_step = 1000
    p.half_life_steps = 100
    p.min = 0.1
    decay = p.Instantiate()
    with self.session():
      with py_utils.GlobalStepContext(0):
        self.assertAllClose(decay.Value().eval(), 1.0)
      with py_utils.GlobalStepContext(500):
        self.assertAllClose(decay.Value().eval(), 1.0)
      with py_utils.GlobalStepContext(1000):
        self.assertAllClose(decay.Value().eval(), 1.0)
      with py_utils.GlobalStepContext(1100):
        self.assertAllClose(decay.Value().eval(), 0.5)
      with py_utils.GlobalStepContext(1200):
        self.assertAllClose(decay.Value().eval(), 0.25)
      with py_utils.GlobalStepContext(1300):
        self.assertAllClose(decay.Value().eval(), 0.125)
      with py_utils.GlobalStepContext(1400):
        self.assertAllClose(decay.Value().eval(), 0.1)
      with py_utils.GlobalStepContext(2000):
        self.assertAllClose(decay.Value().eval(), 0.1)

      # Tests that the decay consistently decreases by half per 100
      # steps.
      for step in range(1000, 1200, 25):
        with py_utils.GlobalStepContext(step):
          a = decay.Value().eval()
        with py_utils.GlobalStepContext(step + 10):
          b = decay.Value().eval()
        with py_utils.GlobalStepContext(step + 100):
          c = decay.Value().eval()
        self.assertGreater(a, b)
        self.assertAllClose(a, c * 2.)

  def testContinuousSchedule_CanOverrideStart(self):
    p = schedule.ContinuousSchedule.Params()
    p.initial_value = 2.0
    p.start_step = 1000
    p.half_life_steps = 100
    decay = p.Instantiate()
    with self.session():
      with py_utils.GlobalStepContext(0):
        self.assertAllClose(decay.Value().eval(), 2.0)
      with py_utils.GlobalStepContext(1000):
        self.assertAllClose(decay.Value().eval(), 2.0)
      with py_utils.GlobalStepContext(1100):
        self.assertAllClose(decay.Value().eval(), 1.0)
      with py_utils.GlobalStepContext(1200):
        self.assertAllClose(decay.Value().eval(), 0.5)
      with py_utils.GlobalStepContext(1300):
        self.assertAllClose(decay.Value().eval(), 0.25)

  def testStepwiseExponentialSchedule(self):
    p = schedule.StepwiseExponentialSchedule.Params()
    p.decay = 0.5
    p.num_steps_per_decay = 1000
    decay = p.Instantiate()
    with self.session():
      with py_utils.GlobalStepContext(0):
        self.assertAllClose(decay.Value().eval(), 1.0)
      with py_utils.GlobalStepContext(999):
        self.assertAllClose(decay.Value().eval(), 1.0)
      with py_utils.GlobalStepContext(1000):
        self.assertAllClose(decay.Value().eval(), 0.5)
      with py_utils.GlobalStepContext(1999):
        self.assertAllClose(decay.Value().eval(), 0.5)
      with py_utils.GlobalStepContext(2000):
        self.assertAllClose(decay.Value().eval(), 0.25)

  def testTransformerSchedule(self):
    p = schedule.TransformerSchedule.Params()
    p.warmup_steps = 4000
    p.model_dim = 512
    lrs = p.Instantiate()
    with self.session():
      expected = [
          1.74693e-07, 0.000174867, 0.00034956, 0.000524253, 0.000698684,
          0.000658735, 0.000624937
      ]
      values = []
      for step in (0, 1000, 2000, 3000, 4000, 4500, 5000):
        with py_utils.GlobalStepContext(step):
          values.append(lrs.Value().eval())
      tf.logging.info('%r' % expected)
      self.assertAllClose(expected, values)

      # Tests that the schedule peaks at 4000 steps.
      with py_utils.GlobalStepContext(3990):
        a = lrs.Value().eval()
      with py_utils.GlobalStepContext(4000):
        b = lrs.Value().eval()
      with py_utils.GlobalStepContext(4010):
        c = lrs.Value().eval()
      self.assertGreater(b, a)
      self.assertGreater(b, c)

      # Tests that the schedule increases linearly before 4000 steps.
      for step in range(300, 4000, 200):
        with py_utils.GlobalStepContext(step - 10):
          a = lrs.Value().eval()
        with py_utils.GlobalStepContext(step):
          b = lrs.Value().eval()
        with py_utils.GlobalStepContext(step + 10):
          c = lrs.Value().eval()
        self.assertAllClose(b * 2., a + c)

  def testTransformerSchedule_CustomDecayFactor(self):
    p = schedule.TransformerSchedule.Params()
    p.warmup_steps = 4000
    p.model_dim = 512
    p.decay_factor = -0.8
    lrs = p.Instantiate()
    with self.session():
      expected = [
          1.450966e-08, 1.452417e-05, 2.903383e-05, 4.354349e-05, 5.802700e-05,
          5.281020e-05, 4.854221e-05
      ]
      values = []
      for step in (0, 1000, 2000, 3000, 4000, 4500, 5000):
        with py_utils.GlobalStepContext(step):
          values.append(lrs.Value().eval())
      tf.logging.info('%r' % expected)
      self.assertAllClose(expected, values)

      # Tests that the schedule peaks at 4000 steps.
      with py_utils.GlobalStepContext(3990):
        a = lrs.Value().eval()
      with py_utils.GlobalStepContext(4000):
        b = lrs.Value().eval()
      with py_utils.GlobalStepContext(4010):
        c = lrs.Value().eval()
      self.assertGreater(b, a)
      self.assertGreater(b, c)

      # Tests that the schedule increases linearly before 4000 steps.
      for step in range(300, 4000, 200):
        with py_utils.GlobalStepContext(step - 10):
          a = lrs.Value().eval()
        with py_utils.GlobalStepContext(step):
          b = lrs.Value().eval()
        with py_utils.GlobalStepContext(step + 10):
          c = lrs.Value().eval()
        self.assertAllClose(b * 2., a + c)

  def testTransformerScheduleWithDecayEnd(self):
    p = schedule.TransformerSchedule.Params()
    p.warmup_steps = 4000
    p.model_dim = 512
    p.decay_end = 5000
    lrs = p.Instantiate()
    with self.session():
      with py_utils.GlobalStepContext(0):
        self.assertAllClose(lrs.Value().eval(), 1.74693e-07)
      with py_utils.GlobalStepContext(3000):
        self.assertAllClose(lrs.Value().eval(), 0.000524253)
      with py_utils.GlobalStepContext(5000):
        self.assertAllClose(lrs.Value().eval(), 0.000624937)

      # Tests that the schedule peaks at 4000 steps.
      with py_utils.GlobalStepContext(3990):
        a = lrs.Value().eval()
      with py_utils.GlobalStepContext(4000):
        b = lrs.Value().eval()
      with py_utils.GlobalStepContext(4010):
        c = lrs.Value().eval()
      self.assertGreater(b, a)
      self.assertGreater(b, c)

      # Tests that the schedule increases linearly before 4000 steps.
      for step in range(300, 4000, 200):
        with py_utils.GlobalStepContext(step - 10):
          a = lrs.Value().eval()
        with py_utils.GlobalStepContext(step):
          b = lrs.Value().eval()
        with py_utils.GlobalStepContext(step + 10):
          c = lrs.Value().eval()
        self.assertAllClose(b * 2., a + c)

      # Tests that the schedule is fixed after decay end steps.
      with py_utils.GlobalStepContext(5000):
        a = lrs.Value().eval()
      with py_utils.GlobalStepContext(4999):
        self.assertGreater(lrs.Value().eval(), a)
      with py_utils.GlobalStepContext(5001):
        self.assertAllClose(lrs.Value().eval(), a)
      with py_utils.GlobalStepContext(6000):
        self.assertAllClose(lrs.Value().eval(), a)

  def testTransformerScheduleNoWarmUp(self):
    params = schedule.TransformerScheduleNoWarmUp.Params().Set(
        decay_start=4000, model_dim=512)
    lrs = params.Instantiate()

    base_params = schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    base_lrs = base_params.Instantiate()

    with self.session():

      # Tests that the schedule is flat up until 4000 steps.
      for step in (0, 1000, 2000, 3000, 4000):
        with py_utils.GlobalStepContext(step):
          self.assertAllClose(lrs.Value().eval(), 0.000698684)
      with py_utils.GlobalStepContext(4500):
        self.assertAllClose(lrs.Value().eval(), 0.000658735)
      with py_utils.GlobalStepContext(5000):
        self.assertAllClose(lrs.Value().eval(), 0.000624937)

      # Test that the schedule is identical with transformer-lr after 4k steps
      for step in (4000, 4010, 5000):
        with py_utils.GlobalStepContext(step):
          self.assertAllClose(base_lrs.Value().eval(), lrs.Value().eval())
          self.assertAllClose(base_lrs.Value().eval(), lrs.Value().eval())
          self.assertAllClose(base_lrs.Value().eval(), lrs.Value().eval())

  def testTransformerMLPerfSchedule(self):
    params = schedule.TransformerMLPerfSchedule.Params().Set(
        warmup_steps=4000, warmup_init_fraction=.3, model_dim=512)
    lrs = params.Instantiate()

    base_params = schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    base_lrs = base_params.Instantiate()

    with self.session():

      # Linear warmup starting from 0.3 * peak_lr.
      peak_lr = 0.000698684
      for step in (0, 1000, 2000, 3000, 4000):
        with py_utils.GlobalStepContext(step):
          self.assertAllClose(.3 * peak_lr + .7 * base_lrs.Value().eval(),
                              lrs.Value().eval())

      # Test that the schedule is identical with transformer-lr after 4k steps
      for step in (4000, 4010, 5000):
        with py_utils.GlobalStepContext(step):
          self.assertAllClose(base_lrs.Value().eval(), lrs.Value().eval())
          self.assertAllClose(base_lrs.Value().eval(), lrs.Value().eval())
          self.assertAllClose(base_lrs.Value().eval(), lrs.Value().eval())

  def testPolynomialLRSchedule(self):
    p = schedule.PolynomialSchedule.Params().Set(
        power=2, start=(0, 0.), limit=(20000, 2.))
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in (0, 10000, 20000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(
          pts,
          [
              [0, 0.0],
              [10000, 0.5],  # 2 * (0.5 ** 2)
              [20000, 2.0],
          ])
      with py_utils.GlobalStepContext(42):
        self.assertEqual(len(lrs.Value().shape), 0)

  def testPolynomialLimitOriginLRSchedule(self):
    p = schedule.PolynomialSchedule.Params().Set(
        power=2, start=(0, 0.), limit=(20000, 2.), origin='limit')
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in (0, 5000, 10000, 15000, 20000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(
          pts,
          [
              [0, 0.0],
              [5000, 0.875],  # 2 * (1 - (1 - 0.25) ** 2)
              [10000, 1.5],  # 2 * (1 - (1 - 0.5) ** 2)
              [15000, 1.875],  # 2 * (1 - (1 - 0.75) ** 2)
              [20000, 2.0],
          ])
      with py_utils.GlobalStepContext(42):
        self.assertEqual(len(lrs.Value().shape), 0)

  def testCombinedLRSchedule(self):
    p = schedule.CombinedMinimumSchedule.Params().Set(schedules=[
        schedule.LinearSchedule.Params().Set(
            start=(0., 1.), limit=(2000000, 8.)),
        schedule.LinearSchedule.Params().Set(
            start=(2000000., 8.), limit=(4000000, 8.)),
        schedule.ExponentialSchedule.Params().Set(
            start=(4000000., 8.), limit=(8000000, 0.5))
    ])
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 10000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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
    p = schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=250000, decay_start=32000000, decay_end=64000000, min=0.5)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 10000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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
    p = schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup_init=0,
        warmup=250000,
        decay_start=32000000,
        decay_end=64000000,
        min=0.5)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 10000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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
    p = schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=250000,
        decay_start=32000000,
        decay_end=64000000,
        min=0.5,
        max=5.0)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 10000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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
    p = schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
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
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 10000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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

  def testLinearRampupExponentialDecayScaledByNumSplitScheduleNoWarmUp(self):
    p = schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=0, decay_start=32000000, decay_end=64000000, min=0.5)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 10000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(
          pts,
          [
              # Constant
              [0, 8.0],
              [1000000, 8.0],
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

  def testLinearRampupExponentialDecayScaledByNumSplitScheduleExpOnly(self):
    p = schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    ).Set(
        warmup=0, decay_start=0, decay_end=32000000, min=0.5)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=8):
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 6000000, 1000000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(
          pts,
          [
              # Exponentially decreasing.
              [0, 8.0],
              [1000000, 4.0],
              [2000000, 2.0],
              [3000000, 1.0],
              [4000000, 0.5],
              [5000000, 0.5]
          ])

  def testLinearRampupSqrtDecayByBatchSizeAndReplicasSchedule(self):
    p = schedule.LinearRampupSqrtDecayByBatchSizeAndReplicas.Params().Set(
        warmup_examples=100000, batch_size=100)
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', gpus=10):
      lrs = p.Instantiate()
      with py_utils.GlobalStepContext(-1):
        self.assertAllClose(lrs.Value().eval(), 0.0)
      with py_utils.GlobalStepContext(49):
        self.assertAllClose(lrs.Value().eval(), 0.05)
      with py_utils.GlobalStepContext(99):
        self.assertAllClose(lrs.Value().eval(), 0.1)
      with py_utils.GlobalStepContext(399):
        self.assertAllClose(lrs.Value().eval(), 0.05)
      with py_utils.GlobalStepContext(1599):
        self.assertAllClose(lrs.Value().eval(), 0.025)

  def testDevBasedSchedule(self):
    logdir = tf.test.get_temp_dir()
    tf.io.gfile.mkdir(os.path.join(logdir, 'eval_dev'))

    p = schedule.DevBasedSchedule.Params()
    p.tolerance = 1.0
    p.window = 2
    p.decay = 0.5
    p.min_factor = 0.20
    early_stop.MetricHistory.SetLogdirInMetricHistories(p, logdir)

    lrs = p.Instantiate()
    self.assertEqual(lrs.theta.cur_factor.name, 'LRSched/cur_factor/var:0')
    self.assertEqual(lrs.theta.ref_step.name, 'LRSched/ref_step/var:0')

    mh = lrs._metric_history
    mh.params.local_filesystem = True
    with self.session():
      self.evaluate(tf.global_variables_initializer())
      with py_utils.GlobalStepContext(0):
        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 1, 10.0)
        # best = 1
        self.assertAllClose(lrs.Value().eval(), 1.0)

        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 2, 5.0)
        # best = 2
        self.assertAllClose(lrs.Value().eval(), 1.0)

        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 5, 4.0)
        # best = 2, out of window
        self.assertAllClose(lrs.Value().eval(), 0.5)

        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 6, 4.0)
        # best = 2, ref = 5, in window
        self.assertAllClose(lrs.Value().eval(), 0.5)

        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 9, 4.0)
        # best = 2, ref = 5, out of window
        self.assertAllClose(lrs.Value().eval(), 0.25)

        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 10, 3.9)
        # best = 10
        self.assertAllClose(lrs.Value().eval(), 0.25)

        mh.ConditionalAppend(mh.params.jobname, mh.params.metric, 13, 3.0)
        # best = 10, out of window, min factor
        self.assertAllClose(lrs.Value().eval(), 0.20)

  def testLinearRampupPiecewiseConstantSchedule(self):
    p = schedule.LinearRampupPiecewiseConstantSchedule.Params().Set(
        boundaries=[40, 64, 80, 96],
        lrs=[1.0, 0.1, 0.01, 0.001],
    )
    with self.session(), cluster_factory.ForTestingWorker(
        mode='sync', job='trainer_client', tpus=8):
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 15, 1):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])

      self.assertAllClose(
          pts, [[0, 0.0], [1, 1.6], [2, 3.2], [3, 4.8], [4, 6.4], [5, 8.0],
                [6, 8.0], [7, 8.0], [8, 0.8], [9, 0.8], [10, 0.08], [11, 0.08],
                [12, 0.008], [13, 0.008], [14, 0.008]])

  def testCosineSchedule(self):
    p = schedule.CosineSchedule.Params().Set(
        initial_value=3.0, final_value=1.0, total_steps=400000)
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 600000, 100000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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
    p = schedule.LinearRampupCosineSchedule.Params().Set(
        warmup_steps=200,
        initial_value=3.0,
        final_value=1.0,
        total_steps=400000,
        num_splits=1)
    with self.session():
      lrs = p.Instantiate()

      pts = []
      for step in [0, 100, 200, 100000, 200000, 300000, 400000]:
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(
          pts,
          [
              [0, 0.0],
              [100, 1.5],
              [200, 3.0],
              [100000, math.cos(math.pi / 4) + 2.],  # angle=pi/4
              [200000, 2.0],  # angle=pi/2, half-way
              [300000, math.cos(math.pi * 3 / 4) + 2.],  # angle=pi*3/4
              [400000, 1.0],
          ])

  def testPiecewiseSchedule(self):
    # Linear ramp-up in 20000 steps, cosine decay in 40000 steps.
    p0 = schedule.LinearSchedule.Params().Set(start=(0, 0.), limit=(20000, 2.))
    p1 = schedule.CosineSchedule.Params().Set(
        initial_value=2.0, total_steps=40000)
    p = schedule.PiecewiseSchedule.Params().Set(
        boundaries=[20000], schedules=[p0, p1])
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 70000, 10000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
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

  def testCycleSchedule(self):
    p0 = schedule.LinearSchedule.Params().Set(start=(0, 0.), limit=(1000, 1.))
    p1 = schedule.Constant.Params().Set(value=5.0)
    p = schedule.CycleSchedule.Params().Set(schedules=[p0, p1], steps=[4, 1])
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in [0, 1, 4, 5, 998, 999, 1000]:
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(pts, [
          [0, 0.0],
          [1, 1.0 / 1000.0],
          [4, 5.0],
          [5, 5.0 / 1000.0],
          [998, 998.0 / 1000.0],
          [999, 5.0],
          [1000, 1.0],
      ])

  def testAnnealingSchedule(self):
    p = schedule.AnnealingSchedule.Params().Set(
        init=1, lower_bound=0.5, factor=0.8)
    lrs = p.Instantiate()

    with self.session():
      pts = []
      for step in range(5):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(pts, [
          [0, 1.0],
          [1, 0.8],
          [2, 0.640],
          [3, 0.512],
          [4, 0.5],
      ])

  def testInverseSigmoid(self):
    p = schedule.InverseSigmoid.Params().Set(k=10000)
    with self.session():
      lrs = p.Instantiate()
      pts = []
      for step in range(0, 200000, 25000):
        with py_utils.GlobalStepContext(step):
          pts.append([step, lrs.Value().eval()])
      self.assertAllClose(
          [[0, 0.999900], [25000, 0.998783], [50000, 0.985376],
           [75000, 0.846880], [100000, 0.312242], [125000, 0.035928],
           [150000, 0.003050], [175000, 0.000251]], pts)


if __name__ == '__main__':
  tf.test.main()
