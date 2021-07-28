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
      l.TrackQTensor('test')
      fns = l.fns

      # Just testing one dynamic and one const op.
      # Dynamic.
      fns.qadd(1, 1, qt='test')
      fns.qadd(1, 1, qmin=-1.0, qmax=1.0)
      with self.assertRaises(AssertionError):
        fns.qadd(1, 1)  # No range args.
      with self.assertRaises(AssertionError):
        fns.qadd(1, 1, qmin=-1.0)  # Incomplete range args.
      with self.assertRaises(AssertionError):
        fns.qadd(1, 1, qmax=-1.0)  # Incomplete range args.
      with self.assertRaisesRegex(AssertionError, 'first calling TrackQTensor'):
        fns.qadd(1, 1, qt='non_existing')  # Test that qt is resolved.

      # Const.
      fns.qtanh(6.0)  # No min/max.
      fns.qtanh(6.0, qmin=-5.0, qmax=6.0)  # Min/max
      fns.qtanh(6.0, qt='test')
      with self.assertRaisesRegex(AssertionError, 'first calling TrackQTensor'):
        fns.qtanh(6.0, qt='non_existing')  # Test that qt has precedence.

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
      l = self._testLayerHelper(
          'testLayerWithPassiveAsymQDomain', p, expected=expected)
      init_minmax_vars = l.qdomain_default._qvars.Transform(lambda x: x.eval())
      print('Initial Minmax vars:', init_minmax_vars)
      # Record.
      with py_utils.GlobalStepContext(16):
        self.evaluate([l.PostTrainingStepUpdate()])
      minmax_vars = l.qdomain_default._qvars.Transform(lambda x: x.eval())
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


class ClippingCapScheduleTest:

  def testLinearClippingCapSchedule(self):
    p = quant_utils.LinearClippingCapSchedule.Params()
    p.start_step = 50
    p.end_step = 100
    p.start_cap = 6.0
    p.end_cap = 1.0
    cc_schedule = p.Instantiate()
    with self.session():
      self.assertAllClose(cc_schedule._Value(25).eval(), 6.0)
      self.assertAllClose(cc_schedule._Value(50).eval(), 6.0)
      self.assertAllClose(cc_schedule._Value(60).eval(), 5.0)
      self.assertAllClose(cc_schedule._Value(70).eval(), 4.0)
      self.assertAllClose(cc_schedule._Value(80).eval(), 3.0)
      self.assertAllClose(cc_schedule._Value(90).eval(), 2.0)
      self.assertAllClose(cc_schedule._Value(100).eval(), 1.0)
      self.assertAllClose(cc_schedule._Value(110).eval(), 1.0)

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
