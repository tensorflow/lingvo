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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function

from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import test_utils


class SampleQuantizedProjectionLayer(quant_utils.QuantizableLayer):
  """Simple projection layer to demonstrate quantization."""

  @classmethod
  def Params(cls):
    p = super(SampleQuantizedProjectionLayer, cls).Params()
    p.Define('input_dim', 2, 'Depth of the input.')
    p.Define('output_dim', 3, 'Depth of the output.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(SampleQuantizedProjectionLayer, self).__init__(params)
    p = self.params

    w_pc = py_utils.WeightParams(
        shape=[p.input_dim, p.output_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('w', w_pc)

    self.TrackQTensor('inputs', 'transformed')

  def FProp(self, theta, inputs, paddings):
    p = self.params
    fns = self.fns

    # It is the most important that weights and top-level activations
    # be tagged for quantization:
    #   - Weights use the self.QWeight() decorator
    #   - Inputs/activations are decorated with self.QTensor(). In general,
    #     the provided name should match a call to self.TrackQTensor in the
    #     constructor. This creates an tensor that is individually accounted
    #     for.
    w = fns.qweight(theta.w)
    inputs = self.QTensor('inputs', inputs)

    # Note the use of the qmatmul from the function library. This will
    # automatically track the output against the qtensor 'transformed'.
    out = fns.qmatmul(
        tf.reshape(inputs, [-1, p.input_dim]), w, qt='transformed')
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], [p.output_dim]], 0))

    # Decorate outputs of simple activation functions with their corresponding
    # range decorator. This will ensure that the result does not exceed the
    # precision of the underlying representation.
    out = fns.qtanh(out)

    # Perform padding manipulation via booleans instead of:
    #   out *= 1.0 - paddings
    # Because the paddings can exist in entirely different numeric ranges than
    # the tensor they are being applied to, it is best to not perform
    # arithmetic directly between them. Instead, broadcast them to the needed
    # size (if different) and perform an exact mask with tf.where.
    # For added numeric range protection, the QRPadding decorator ensures
    # the correct range. This is mostly needed for cases where padding is
    # dynamic at inference time.
    paddings = self.QRPadding(paddings)
    paddings *= tf.ones_like(out)  # Broadcast to 'out' size.
    out = tf.where(paddings > 0.0, tf.zeros_like(out), out)

    return out


class QuantizableLayerTest(test_utils.TestCase):
  # pyformat: disable
  NO_QDOMAIN_EXPECTED = [
   [[ 0.00071405, -0.03868543, -0.01999986, -0.00994987],
    [ 0.08905827,  0.13636404, -0.03180931,  0.06056439],
    [ 0.        ,  0.        ,  0.        ,  0.        ],
    [-0.0208858 , -0.17595209, -0.05192588,  0.02618068]],
   [[ 0.        ,  0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        ],
    [-0.02125708, -0.10454545, -0.01147466,  0.06903321],
    [ 0.0276652 , -0.14823943, -0.09726462,  0.01415125]]]
  # pyformat: enable

  def testOpWrapperArgChecking(self):
    with self.session():
      p = SampleQuantizedProjectionLayer.Params()
      p.name = 'test'
      l = p.cls(p)
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
      with self.assertRaisesRegexp(AssertionError,
                                   'first calling TrackQTensor'):
        fns.qadd(1, 1, qt='non_existing')  # Test that qt is resolved.

      # Const.
      fns.qtanh(6.0)  # No min/max.
      fns.qtanh(6.0, qmin=-5.0, qmax=6.0)  # Min/max
      fns.qtanh(6.0, qt='test')
      with self.assertRaisesRegexp(AssertionError,
                                   'first calling TrackQTensor'):
        fns.qtanh(6.0, qt='non_existing')  # Test that qt has precedence.

  def testLayerWithNoQDomain(self):
    with self.session() as sess:
      p = SampleQuantizedProjectionLayer.Params()
      self._testLayerHelper('testLayerWithNoQDomain', sess, p,
                            self.NO_QDOMAIN_EXPECTED)

  def testLayerWithIdentityQDomain(self):
    with self.session() as sess:
      p = SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.QDomain.Params()
      self._testLayerHelper('testLayerWithIdentityQDomain', sess, p,
                            self.NO_QDOMAIN_EXPECTED)

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

    with self.session() as sess:
      p = SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
      l = self._testLayerHelper(
          'testLayerWithPassiveAsymQDomain', sess, p, expected=expected)
      init_minmax_vars = l.qdomain_default._qvars.Transform(lambda x: x.eval())
      print('Initial Minmax vars:', init_minmax_vars)
      # Record.
      sess.run([l.PostTrainingStepUpdate(16)])
      minmax_vars = l.qdomain_default._qvars.Transform(lambda x: x.eval())
      print('Minmax vars:', minmax_vars)

      # Make sure that the vars have moved from their defaults.
      for k in minmax_vars:
        self.assertNotEqual(init_minmax_vars[k], minmax_vars[k])

  def testLayerWithPassiveAsymQDomainTrainQuantDisabledInital(self):
    p = SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = -1
    with self.session() as sess:
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDisabledInital',
          sess,
          p,
          expected=self.NO_QDOMAIN_EXPECTED)

  def testLayerWithPassiveAsymQDomainTrainQuantDisabledStep16(self):
    p = SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = -1
    with self.session() as sess:
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDisabledStep16',
          sess,
          p,
          expected=self.NO_QDOMAIN_EXPECTED,
          global_step=16)

  def testLayerWithPassiveAsymQDomainEvalQuantDisabled(self):
    p = SampleQuantizedProjectionLayer.Params()
    p.is_eval = True
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = -1
    with self.session() as sess:
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainEvalQuantDisabled',
          sess,
          p,
          not_expected=self.NO_QDOMAIN_EXPECTED)

  def testLayerWithPassiveAsymQDomainTrainQuantDelayNotSatisfied(self):
    p = SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = 8
    with self.session() as sess:
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDelayNotSatisfied',
          sess,
          p,
          expected=self.NO_QDOMAIN_EXPECTED,
          global_step=3)

  def testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfied(self):
    p = SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = 8
    with self.session() as sess:
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfied',
          sess,
          p,
          not_expected=self.NO_QDOMAIN_EXPECTED,
          global_step=8)

  def testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfiedPlusOne(self):
    p = SampleQuantizedProjectionLayer.Params()
    p.qdomain.default = quant_utils.PassiveAsymQDomain.Params()
    p.qdomain.default.delay_start_steps = 8
    with self.session() as sess:
      self._testLayerHelper(
          'testLayerWithPassiveAsymQDomainTrainQuantDelaySatisfied',
          sess,
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

    with self.session() as sess:
      p = SampleQuantizedProjectionLayer.Params()
      p.qdomain.default = quant_utils.SymmetricScheduledClipQDomain.Params()
      p.qdomain.default.cc_schedule.Set(
          clip_start_step=0,
          clip_end_step=5,
          quant_start_step=10,
      )
      self._testLayerHelper(
          'testLayerWithSymmetricScheduledClipQDomain',
          sess,
          p,
          expected=expected,
          global_step=16)

  def _testLayerHelper(self,
                       test_case,
                       sess,
                       p,
                       expected=None,
                       not_expected=None,
                       global_step=-1):
    tf.set_random_seed(398847392)
    np.random.seed(12345)
    p.name = 'proj'
    p.input_dim = 3
    p.output_dim = 4
    p.params_init = py_utils.WeightInit.Gaussian(0.1)
    l = p.cls(p)
    in_padding = tf.zeros([2, 4, 1], dtype=tf.float32)
    in_padding = tf.constant(
        [[[0], [0], [1], [0]], [[1], [1], [0], [0]]], dtype=tf.float32)
    inputs = tf.constant(
        np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)
    output = l.FPropDefaultTheta(inputs, in_padding)
    tf.global_variables_initializer().run()

    if global_step >= 0:
      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), global_step))

    output = output.eval()
    print('QuantizableLayerTest output', test_case, ':\n',
          np.array_repr(output))
    if expected is not None:
      self.assertAllClose(output, expected)
    if not_expected is not None:
      self.assertNotAllClose(output, not_expected)
    return l


class ClippingCapScheduleTest(object):

  def testLinearClippingCapSchedule(self):
    p = quant_utils.LinearClippingCapSchedule.Params()
    p.start_step = 50
    p.end_step = 100
    p.start_cap = 6.0
    p.end_cap = 1.0
    cc_schedule = p.cls(p)
    with self.session():
      print(cc_schedule.Value(25).eval())
      print(cc_schedule.Value(50).eval())
      print(cc_schedule.Value(60).eval())
      print(cc_schedule.Value(70).eval())
      print(cc_schedule.Value(80).eval())
      print(cc_schedule.Value(90).eval())
      print(cc_schedule.Value(100).eval())
      print(cc_schedule.Value(110).eval())
      self.assertAllClose(cc_schedule.Value(25).eval(), 6.0)
      self.assertAllClose(cc_schedule.Value(50).eval(), 6.0)
      self.assertAllClose(cc_schedule.Value(60).eval(), 5.0)
      self.assertAllClose(cc_schedule.Value(70).eval(), 4.0)
      self.assertAllClose(cc_schedule.Value(80).eval(), 3.0)
      self.assertAllClose(cc_schedule.Value(90).eval(), 2.0)
      self.assertAllClose(cc_schedule.Value(100).eval(), 1.0)
      self.assertAllClose(cc_schedule.Value(110).eval(), 1.0)

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
    with self.session() as sess:
      cc_schedule = p.cls(p)
      tf.global_variables_initializer().run()
      # Move to fully quantized part of schedule
      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 16))

      @function.Defun(tf.float32, tf.float32)
      def ExampleFunction8(x, cc_state):
        return cc_schedule.ApplyClippingWithState(cc_state, x, bits=8)

      @function.Defun(tf.float32, tf.float32)
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
    with self.session() as sess:
      cc_schedule = p.cls(p)
      tf.global_variables_initializer().run()
      # Step 0: No clipping.
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0), (-100.0, 100.0))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 5: Clipping active but not yet quantizing.
      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 5))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0),
          (-6.0, 5.953125))  # 6 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 7: Middle of clipping range.
      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 7))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0), (-4.0, 3.96875))  # 4 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 10: End of clipping range.
      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 10))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0),
          (-1.0, 0.9921875))  # 1 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 11: No more clipping but not yet quantizing.
      sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), 11))
      self.assertAllClose(
          self._ClipExample(cc_schedule, 100.0),
          (-1.0, 0.9921875))  # 1 * 127/128
      self.assertAllClose(
          self._ClipExample(cc_schedule, 0.123456),
          (-0.123456, 0.123456))  # Not Quantized.

      # Step 15-16: Quantizing at full clip.
      for step in (15, 16):
        sess.run(tf.assign(py_utils.GetOrCreateGlobalStepVar(), step))
        self.assertAllClose(
            self._ClipExample(cc_schedule, 100.0),
            (-1.0, 0.9921875))  # 1 * 127/128
        self.assertAllClose(
            self._ClipExample(cc_schedule, 0.123456),
            (-0.125, 0.125))  # Quantized.


if __name__ == '__main__':
  tf.test.main()
