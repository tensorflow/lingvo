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
"""Tests for learning rate schedules."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax import test_util
from lingvo.core import py_utils as tf_py_utils
from lingvo.core import schedule as tf_schedule
from lingvo.jax import schedules


class SchedulesTest(test_util.JaxTestCase):

  @parameterized.parameters((0,), (10,), (100,), (1000000,))
  def test_constant_schedule(self, count):
    lr_value = 5.
    p = schedules.ConstantSchedule.Params().Set(value=lr_value)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    with self.subTest(name='reference_values'):
      self.assertAllClose(jit_value(count), lr_value)

    tf_p = tf_schedule.Constant.Params().Set(value=lr_value)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      with tf_py_utils.GlobalStepContext(count):
        self.assertAllClose(
            jit_value(jnp.array(count)),
            tf_lr_schedule.Value().numpy())

  @parameterized.parameters((29, 1.), (39, 0.1), (49, 0.01), (50, 0.001),
                            (59, 0.001))
  def test_piecewise_constant_schedule(self, count, expected_value):
    boundaries = [30, 40, 50]
    values = [1.0, 0.1, 0.01, 0.001]
    p = schedules.PiecewiseConstantSchedule.Params().Set(
        boundaries=boundaries, values=values)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    with self.subTest(name='reference_values'):
      self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.PiecewiseConstantSchedule.Params().Set(
        boundaries=boundaries, values=values)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      with tf_py_utils.GlobalStepContext(count):
        self.assertAllClose(
            jit_value(jnp.array(count)),
            tf_lr_schedule.Value().numpy())

  @parameterized.parameters(
      (0, 1), (10, 1), (100, 1), (1000000, 1), (0, 2), (10, 2), (100, 2),
      (1000000, 2), (0, 3), (10, 3), (100, 3), (1000000, 3), (0, 4), (10, 4),
      (100, 4), (1000000, 4), (0, 5), (10, 5), (100, 5), (1000000, 5))
  def test_polynomial_schedule(self, count, power):
    p = schedules.PolynomialSchedule.Params().Set(
        start=(7, 0.9), limit=(370, 1.3), power=power)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    tf_p = tf_schedule.PolynomialSchedule.Params().Set(
        start=(7, 0.9), limit=(370, 1.3), power=power)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      with tf_py_utils.GlobalStepContext(count):
        self.assertAllClose(
            jit_value(jnp.array(count)),
            tf_lr_schedule.Value().numpy())

  @parameterized.parameters((0, 1.74693e-07), (1000, 0.000174867),
                            (2000, 0.00034956), (3000, 0.000524253),
                            (4000, 0.000698684), (4500, 0.000658735),
                            (5000, 0.000624937))
  def test_transformer_schedule_values(self, count, expected_value):
    count = jnp.array(count)
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    with self.subTest(name='reference_values'):
      self.assertAllClose(jit_value(count), expected_value)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      with tf_py_utils.GlobalStepContext(count):
        self.assertAllClose(
            jit_value(jnp.array(count)),
            tf_lr_schedule.Value().numpy())

  def test_transformer_schedule_peak(self):
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    # Tests that the schedule peaks at 4000 steps.
    v_3990 = jit_value(jnp.array(3990))
    v_4000 = jit_value(jnp.array(4000))
    v_4010 = jit_value(jnp.array(4010))
    with self.subTest(name='reference_values'):
      self.assertGreater(v_4000, v_3990)
      self.assertGreater(v_4000, v_4010)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in {3990, 4000, 4010}:
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_transformer_schedule_linear(self):
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    # Tests that the schedule increases linearly before 4000 steps.
    with self.subTest(name='reference_values'):
      for step in range(300, 4000, 200):
        a = jit_value(jnp.array(step - 10))
        b = jit_value(jnp.array(step))
        c = jit_value(jnp.array(step + 10))
        self.assertAllClose(b * 2., a + c)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in range(300, 4000, 200):
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  @parameterized.parameters((0, 1.74693e-07), (1000, 0.000174867),
                            (2000, 0.00034956), (3000, 0.000524253),
                            (4000, 0.000698684), (4500, 0.000658735),
                            (5000, 0.000624937))
  def test_transformer_schedule_with_decay_end_values(self, count,
                                                      expected_value):
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    with self.subTest(name='reference_values'):
      self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      with tf_py_utils.GlobalStepContext(count):
        self.assertAllClose(
            jit_value(jnp.array(count)),
            tf_lr_schedule.Value().numpy())

  def test_transformer_schedule_with_decay_end_peak(self):
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    with self.subTest(name='reference_values'):
      # Tests that the schedule peaks at 4000 steps.
      v_3990 = jit_value(jnp.array(3990))
      v_4000 = jit_value(jnp.array(4000))
      v_4010 = jit_value(jnp.array(4010))
      self.assertGreater(v_4000, v_3990)
      self.assertGreater(v_4000, v_4010)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in {3990, 4000, 4010}:
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_transformer_schedule_with_decay_end_linear(self):
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    # Tests that the schedule increases linearly before 4000 steps.
    with self.subTest(name='reference_values'):
      for step in range(300, 4000, 200):
        a = jit_value(jnp.array(step - 10))
        b = jit_value(jnp.array(step))
        c = jit_value(jnp.array(step + 10))
        self.assertAllClose(b * 2., a + c)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in range(300, 4000, 200):
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_transformer_schedule_with_decay_end_fixed(self):
    p = schedules.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    # Tests that the schedule is fixed after decay end steps.
    v_decay_end = lr_schedule.Value(jnp.array(p.decay_end))
    with self.subTest(name='reference_values'):
      self.assertGreater(jit_value(jnp.array(p.decay_end - 1)), v_decay_end)
      self.assertAllClose(jit_value(jnp.array(p.decay_end + 1)), v_decay_end)
      self.assertAllClose(jit_value(jnp.array(p.decay_end + 1000)), v_decay_end)

    tf_p = tf_schedule.TransformerSchedule.Params().Set(
        warmup_steps=4000, model_dim=512, decay_end=5000)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in range(p.decay_end - 1, p.decay_end + 20, 2):
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  @parameterized.parameters((0,), (1000,), (2000,), (3000,), (4000,), (4500,),
                            (5000,))
  def test_sqrt_decay_schedule_values(self, count):
    p = schedules.SqrtDecaySchedule.Params().Set(warmup_steps=4000)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    tf_p = tf_schedule.SqrtDecay.Params().Set(warmup_steps=4000)
    tf_lr_schedule = tf_p.Instantiate()
    with tf_py_utils.GlobalStepContext(count):
      self.assertAllClose(
          jit_value(jnp.array(count)),
          tf_lr_schedule.Value().numpy())

  def test_linear_schedule_values(self):
    p = schedules.LinearSchedule.Params().Set(
        start=(100, 0.1), limit=(200, 1.0))
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    xs = [0, 10, 20, 100, 120, 150, 200, 250]
    expected_values = [0.1, 0.1, 0.1, 0.1, 0.28, 0.55, 1.0, 1.0]
    with self.subTest(name='reference_values'):
      for count, expected_value in zip(xs, expected_values):
        self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.LinearSchedule.Params().Set(
        start=(100, 0.1), limit=(200, 1.0))
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in xs:
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_exponential_schedule(self):
    p = schedules.ExponentialSchedule.Params().Set(
        start=(100, 1.0), limit=(200, 0.1))
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    xs = [0, 10, 20, 100, 120, 150, 200, 250]
    expected_values = [1.0, 1.0, 1.0, 1.0, 0.6309573, 0.3162277, 0.1, 0.1]
    with self.subTest(name='reference_values'):
      for count, expected_value in zip(xs, expected_values):
        self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.ExponentialSchedule.Params().Set(
        start=(100, 1.0), limit=(200, 0.1))
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in xs:
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_linear_rampup_exp_decay_schedule(self):
    p = schedules.LinearRampupExponentialDecay.Params().Set(
        warmup=100, decay_start=200, decay_end=300, max=1.0, min_ratio=0.01)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    xs = [0, 10, 20, 100, 120, 150, 200, 250, 300, 350]
    expected_values = [0.0, 0.1, 0.2, 1.0, 1.0, 1.0, 1.0, 0.1, 0.01, 0.01]
    with self.subTest(name='reference_values'):
      for count, expected_value in zip(xs, expected_values):
        self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.LinearRampupExponentialDecay.Params().Set(
        warmup=100, decay_start=200, decay_end=300, max=1.0, min=0.01)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in xs:
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_linear_rampup_exp_decay_schedule_noconstant(self):
    p = schedules.LinearRampupExponentialDecay.Params().Set(
        warmup=150, decay_start=150, decay_end=250, max=1.0, min_ratio=0.01)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    xs = [0, 15, 30, 150, 200, 250, 300, 350]
    expected_values = [0., 0.1, 0.2, 1.0, 0.1, 0.01, 0.01, 0.01]
    with self.subTest(name='reference_values'):
      for count, expected_value in zip(xs, expected_values):
        self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.LinearRampupExponentialDecay.Params().Set(
        warmup=150, decay_start=150, decay_end=250, max=1.0, min=0.01)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in xs:
        if count == 200:
          # Lingvo implementation does not support no warm-up. It just adds a
          # warm-up consisting of a single step. Hence, no comparison.
          continue
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_linear_rampup_exp_decay_schedule_nowarmup(self):
    p = schedules.LinearRampupExponentialDecay.Params().Set(
        warmup=0, decay_start=0, decay_end=100, max=1.0, min_ratio=0.01)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    xs = [0, 50, 100, 150, 200]
    expected_values = [1., 0.1, 0.01, 0.01, 0.01]
    with self.subTest(name='reference_values'):
      for count, expected_value in zip(xs, expected_values):
        self.assertAllClose(jit_value(jnp.array(count)), expected_value)

    tf_p = tf_schedule.LinearRampupExponentialDecay.Params().Set(
        warmup=0, decay_start=0, decay_end=100, max=1.0, min=0.01)
    tf_lr_schedule = tf_p.Instantiate()
    with self.subTest(name='lingvo_values'):
      for count in xs:
        if count == 50:
          # Lingvo implementation does not support no warm-up. It just adds a
          # warm-up consisting of a single step. Hence, no comparison.
          continue
        with tf_py_utils.GlobalStepContext(count):
          self.assertAllClose(
              jit_value(jnp.array(count)),
              tf_lr_schedule.Value().numpy())

  def test_linear_rampup_piecewise_constant_schedule(self):
    boundaries = [40, 64, 80, 96]
    values = [1.0, 0.1, 0.01, 0.001]
    p = schedules.LinearRampupPiecewiseConstantSchedule.Params().Set(
        boundaries=boundaries, values=values)
    lr_schedule = p.Instantiate()
    jit_value = jax.jit(lr_schedule.Value)

    tf_p = tf_schedule.LinearRampupPiecewiseConstantSchedule.Params().Set(
        boundaries=boundaries, lrs=values, num_splits=1)
    tf_lr_schedule = tf_p.Instantiate()
    for step in range(100):
      with tf_py_utils.GlobalStepContext(step):
        self.assertAllClose(
            jit_value(jnp.array(step)),
            tf_lr_schedule.Value().numpy())


if __name__ == '__main__':
  absltest.main()
