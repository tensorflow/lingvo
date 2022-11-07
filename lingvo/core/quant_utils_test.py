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
"""Tests for quant_utils."""

# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation


import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import quant_test_lib
from lingvo.core import quant_utils


class QuantizableLayerTest(quant_test_lib.QuantUtilsBaseTest):

  def testOpWrapperArgChecking(self):
    with self.session():
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      p.name = 'test'
      l = p.Instantiate()
      l.TrackQActs('test')
      fns = l.fns

      # Just testing one dynamic and one const op.
      # Dynamic.
      fns.qadd(1, 1, qout_name='test')
      with self.assertRaises(ValueError):
        fns.qadd(1, 1)  # No qout_name.
      with self.assertRaisesRegex(ValueError, 'TrackQActs'):
        fns.qadd(1, 1, qout_name='non_existing')  # Test qout_name is resolved.

      # Known range tests.
      fns.qtanh(6.0)
      fns.qtanh(6.0, qout_name='test')
      with self.assertRaisesRegex(ValueError, 'TrackQActs'):
        fns.qtanh(6.0, qout_name='non_existing')  # Test qout_name precedence.

  def testLayerWithNoQDomain(self):
    with self.session():
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      self._testLayerHelper('testLayerWithNoQDomain', p,
                            self.NO_QDOMAIN_EXPECTED)

  def testLayerWithIdentityQDomain(self):
    with self.session():
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.QDomain.Params()
      self._testLayerHelper('testLayerWithIdentityQDomain', p,
                            self.NO_QDOMAIN_EXPECTED)

  def testGetQDomainParams(self):
    p = quant_test_lib.SampleQuantizedProjectionLayer.Params().Set(name='test')
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params().Set(bits=3)
    p.qdomain.test_1 = quant_utils.PassiveAsymQDomain.Params().Set(bits=47)
    layer = p.Instantiate()
    self.assertEqual(layer.GetQDomainParams('default').bits, 3)
    self.assertEqual(layer.GetQDomainParams('test_1').bits, 47)
    self.assertEqual(layer.GetQDomainParams('test_2').bits, 3)

  def testLayerWithPassiveAsymQDomain(self):
    # pyformat: disable
    expected = [
       [[ 0.        , -0.03921568, -0.02352941, -0.00784314],
        [ 0.0862745 ,  0.13333333, -0.03137255,  0.06274509],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [-0.02352941, -0.17254901, -0.05490196,  0.02352941]],
       [[ 0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [-0.02352941, -0.10196078, -0.00784314,  0.07058823],
        [ 0.02352941, -0.1490196 , -0.09411764,  0.01568627]]]
    # pyformat: enable

    with self.session():
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
      p.qdomain.default.narrow_to_asym_bit_depth = False
      l = self._testLayerHelper(
          'testLayerWithPassiveAsymQDomain', p, expected=expected)
      init_minmax_vars = {
          k: v.eval() for k, v in l.qdomain_default._qvars.items()
      }
      print('Initial Minmax vars:', init_minmax_vars)
      # Record.
      with py_utils.GlobalStepContext(16):
        self.evaluate([l.PostTrainingStepUpdate()])
      minmax_vars = {k: v.eval() for k, v in l.qdomain_default._qvars.items()}
      print('Minmax vars:', minmax_vars)

      # Make sure that the vars have moved from their defaults.
      for k in minmax_vars:
        self.assertNotEqual(init_minmax_vars[k], minmax_vars[k])

  def testLayerWithPassiveAsymQDomainTrainQuantDisabledInital(self):
    p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = -1
    with self.session():
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDisabledInital',
          p,
          expected=self.NO_QDOMAIN_EXPECTED)

  def testLayerWithPassiveAsymQDomainTrainQuantDisabledStep16(self):
    p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = -1
    with self.session():
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDisabledStep16',
          p,
          expected=self.NO_QDOMAIN_EXPECTED,
          global_step=16)

  def testLayerWithPassiveAsymQDomainEvalQuantDisabled(self):
    with self.session(), self.SetEval(True):
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
      p.qdomain.default.delay_start_steps = -1
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainEvalQuantDisabled',
          p,
          not_expected=self.NO_QDOMAIN_EXPECTED)

  def testLayerWithPassiveAsymQDomainTrainQuantDelayNotSatisfied(self):
    p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = 8
    with self.session():
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDelayNotSatisfied',
          p,
          expected=self.NO_QDOMAIN_EXPECTED,
          global_step=3)

  def testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfied(self):
    p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = 8
    with self.session():
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfied',
          p,
          not_expected=self.NO_QDOMAIN_EXPECTED,
          global_step=8)

  def testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfiedPlusOne(self):
    p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = 8
    with self.session():
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfied',
          p,
          not_expected=self.NO_QDOMAIN_EXPECTED,
          global_step=9)

  def testLayerWithSymmetricScheduledClipQDomain(self):
    # pyformat: disable
    expected = [
       [[ 0.       , -0.0390625, -0.015625 , -0.0078125],
        [ 0.0859375,  0.140625 , -0.0234375,  0.0625   ],
        [ 0.       ,  0.       ,  0.       ,  0.       ],
        [-0.0234375, -0.171875 , -0.0546875,  0.0234375]],
       [[ 0.       ,  0.       ,  0.       ,  0.       ],
        [ 0.       ,  0.       ,  0.       ,  0.       ],
        [-0.0234375, -0.1015625, -0.015625 ,  0.0703125],
        [ 0.       , -0.125    , -0.0625   ,  0.       ]]]
    # pyformat: enable

    with self.session():
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.SymmetricScheduledClipQDomain.Params()
      p.qdomain.default.cc_schedule.Set(
          clip_start_step=0,
          clip_end_step=5,
          quant_start_step=10,
      )
      self._testLayerHelper(
          'testLayerWithSymmetricScheduledClipQDomain',
          p,
          expected=expected,
          global_step=16)

  def testLayerWithFrozenDomain(self):
    with self.session():
      p = quant_test_lib.SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
      p.qdomain.default.freeze = True
      p.qdomain.default.narrow_to_asym_bit_depth = False
      l = self._testLayerHelper('testLayerWithFrozenQDomain', p, expected=None)
      init_minmax_vars = {
          k: v.eval() for k, v in l.qdomain_default._qvars.items()
      }
      # Record.
      with py_utils.GlobalStepContext(16):
        self.evaluate([l.PostTrainingStepUpdate()])
      minmax_vars = {k: v.eval() for k, v in l.qdomain_default._qvars.items()}

      # Make sure that the vars have not moved from their defaults, because the
      # qdomain is frozen.
      for k in minmax_vars:
        self.assertEqual(init_minmax_vars[k], minmax_vars[k])


class FakeQDomainTest(quant_test_lib.QuantUtilsBaseTest):

  def testCustomLogSoftmaxRequiresRange(self):
    qd = quant_utils.PassiveAsymQDomain.Params().Set(
        name='log_softmax_unset').Instantiate()
    with self.assertRaises(ValueError):
      qd.QRAct(tf.ones((3, 3)), quant_utils.QDistribution.LOG_SOFTMAX)

  def testCustomLogSoftmaxRange(self):
    qd = quant_utils.PassiveAsymQDomain.Params().Set(
        log_softmax_range=(-1, 0), name='log_softmax').Instantiate()
    tf.random.set_seed(0)
    probs = tf.random.uniform((8, 10), minval=0, maxval=1, dtype=tf.float32)
    log_probs = tf.math.log(probs)
    qlog_probs = qd.QRAct(log_probs, quant_utils.QDistribution.LOG_SOFTMAX)
    # Test that the minimum value is clipped as expected.
    self.assertEqual(self.evaluate(tf.reduce_min(qlog_probs)), -1)

  def testSoftmaxExact(self):
    qd = quant_utils.PassiveAsymQDomain.Params().Set(
        name='softmax_exact', bits=4,
        narrow_to_asym_bit_depth=False).Instantiate()
    # Create probs in [0, 1] at intervals matching int4 quantization.
    probs = tf.reshape(tf.range(16, dtype=tf.float32) / 15, (4, 4))
    qprobs = qd.QRAct(probs, quant_utils.QDistribution.SOFTMAX)
    # Test that they are quantized as expected (i.e. ~exactly).
    self.assertAllClose(self.evaluate(qprobs), self.evaluate(probs))

  def testNarrowSoftmaxBitDepth(self):
    qd = quant_utils.PassiveAsymQDomain.Params().Set(
        name='narrow_softmax', bits=4,
        narrow_to_asym_bit_depth=True).Instantiate()
    max_prob = 1
    qmax_prob = qd.QRAct(max_prob, quant_utils.QDistribution.SOFTMAX)
    # Test that the max is clipped to the correct range for TFLite.
    self.assertEqual(self.evaluate(qmax_prob), 0.9375)


class ClippingCapScheduleTest(quant_test_lib.QuantUtilsBaseTest):

  def testLinearClippingCapSchedule(self):
    p = quant_utils.LinearClippingCapSchedule.Params()
    p.start_step = 50
    p.end_step = 100
    p.start_cap = 6.0
    p.end_cap = 1.0
    cc_schedule = p.Instantiate()
    step_to_expected_value = {
        25: 6.0,
        50: 6.0,
        60: 5.0,
        70: 4.0,
        80: 3.0,
        90: 2.0,
        100: 1.0,
        110: 1.0,
    }
    with self.session():
      global_step = py_utils.GetGlobalStep()
      for step, expected_value in step_to_expected_value.items():
        tf.assign(global_step, step).eval()
        self.assertAllClose(cc_schedule._Value().eval(), expected_value)

  def _ClipExample(self, cc_schedule, v):
    """Returns a tuple of (neg, pos) for clipped neg/pos values of v."""
    v = float(v)
    clipped = (
        cc_schedule.ApplyClipping(cc_schedule.theta, -v).eval(),
        cc_schedule.ApplyClipping(cc_schedule.theta, v).eval(),
    )
    print('Clipped +-', v, ' ->', clipped)
    return clipped

  def testFakeQuantizationScheduleFromDefun(self):
    p = quant_utils.FakeQuantizationSchedule.Params()
    p.clip_start_step = 5
    p.clip_end_step = 10
    p.quant_start_step = 15
    p.start_cap = 6.0
    p.end_cap = 1.0
    with self.session():
      cc_schedule = p.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      # Move to fully quantized part of schedule
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 16))

      @tf.function(autograph=False)
      def ExampleFunction8(x, cc_state):
        return cc_schedule.ApplyClippingWithState(cc_state, x, bits=8)

      @tf.function(autograph=False)
      def ExampleFunction16(x, cc_state):
        return cc_schedule.ApplyClippingWithState(cc_state, x, bits=16)

      a = tf.constant(1.0)
      b = tf.constant(0.5)

      # 8bit value.
      v = ExampleFunction8(a * b, cc_schedule.GetState(cc_schedule.theta))
      self.assertAllClose(v.eval(), 0.5)

      # 16bit value.
      v = ExampleFunction16(a * b, cc_schedule.GetState(cc_schedule.theta))
      self.assertAllClose(v.eval(), 0.5)

      # An incomplete implementation requires special case gradient logic.
      # This tests it, specifically in a Defun, which caused issues.
      # 8bit gradient.
      g = tf.gradients(
          ExampleFunction8(a * b, cc_schedule.GetState(cc_schedule.theta)),
          [a, b])
      g = [t.eval() for t in g]
      print('Gradient8:', g)
      self.assertAllClose(g, (0.5, 1.0))

      # 16bit gradient.
      g = tf.gradients(
          ExampleFunction16(a * b, cc_schedule.GetState(cc_schedule.theta)),
          [a, b])
      g = [t.eval() for t in g]
      print('Gradient16:', g)
      self.assertAllClose(g, (0.5, 1.0))

  def testFakeQuantizationScheduleTraining(self):
    p = quant_utils.FakeQuantizationSchedule.Params()
    p.clip_start_step = 5
    p.clip_end_step = 10
    p.quant_start_step = 15
    p.start_cap = 6.0
    p.end_cap = 1.0
    with self.session():
      cc_schedule = p.Instantiate()
      self.evaluate(tf.global_variables_initializer())
      # Step 0: No clipping.
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0), (-100.0, 100.0))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 5: Clipping active but not yet quantizing.
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0),
          (-6.0, 5.953125))  # 6 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 7: Middle of clipping range.
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 7))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0), (-4.0, 3.96875))  # 4 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 10: End of clipping range.
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 10))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0),
          (-1.0, 0.9921875))  # 1 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 11: No more clipping but not yet quantizing.
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 11))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0),
          (-1.0, 0.9921875))  # 1 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 15-16: Quantizing at full clip.
      for step in (15, 16):
        self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), step))
        self.assertAllClose(
            self._ClipExample(cc_schedule, 100.0),
            (-1.0, 0.9921875))  # 1 * 127/128
        self.assertAllClose(
            self._ClipExample(cc_schedule, 0.123456),
            (-0.125, 0.125))  # Quantized.


if __name__ == '__main__':
  tf.test.main()
