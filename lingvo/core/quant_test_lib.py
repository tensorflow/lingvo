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
"""Tests lib for quant_utils test."""

# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation


import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import test_utils
import numpy as np


class SampleQuantizedProjectionLayer(quant_utils.QuantizableLayer):
  """Simple projection layer to demonstrate quantization."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', 2, 'Depth of the input.')
    p.Define('output_dim', 3, 'Depth of the output.')
    p.qdomain.Define('test_1', None, 'Dummy qdomain')
    p.qdomain.Define('test_2', None, 'Dummy qdomain')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateAqtWeight(
        'w',
        shape=[p.input_dim, p.output_dim],
        feature_axis=-1,
        legacy_aqt_w_name='aqt_w')

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params

    w_pc = py_utils.WeightParams(
        shape=[p.input_dim, p.output_dim],
        init=p.params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
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

    reshaped_inputs = tf.reshape(inputs, [-1, p.input_dim])
    reshaped_inputs, w = self.ToAqtInputs(
        'w',
        act=reshaped_inputs,
        weight=w,
        w_feature_axis=-1,
        w_expected_scale_shape=(1, p.output_dim))

    # Note the use of the qmatmul from the function library. This will
    # automatically track the output against the qtensor 'transformed'.
    out = fns.qmatmul(reshaped_inputs, w, qt='transformed')
    out = self.FromAqtMatmul('w', out)

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


class QuantUtilsBaseTest(test_utils.TestCase):
  """Base test class for testing quantizable layer."""

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

  def _testLayerHelper(self,
                       test_case,
                       p,
                       expected=None,
                       not_expected=None,
                       global_step=-1):
    tf.random.set_seed(398847392)
    np.random.seed(12345)
    p.name = 'proj'
    p.input_dim = 3
    p.output_dim = 4
    p.params_init = py_utils.WeightInit.Gaussian(0.1)
    l = p.Instantiate()
    in_padding = tf.zeros([2, 4, 1], dtype=tf.float32)
    in_padding = tf.constant(
        [[[0], [0], [1], [0]], [[1], [1], [0], [0]]], dtype=tf.float32)
    inputs = tf.constant(
        np.random.normal(0.1, 0.5, [2, 4, 3]), dtype=tf.float32)
    output = l.FPropDefaultTheta(inputs, in_padding)
    self.evaluate(tf.global_variables_initializer())

    if global_step >= 0:
      self.evaluate(tf.assign(py_utils.GetOrCreateGlobalStepVar(), global_step))

    output = output.eval()
    print('QuantizableLayerTest output', test_case, ':\n',
          np.array_repr(output))
    if expected is not None:
      self.assertAllClose(output, expected)
    if not_expected is not None:
      self.assertNotAllClose(output, not_expected)
    return l
