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
r"""Test code for gshard_layers."""

from lingvo import compat as tf
from lingvo.core import gshard_builder
from lingvo.core import gshard_layers
from lingvo.core import test_utils
import numpy as np

FLAGS = tf.flags.FLAGS


class CausalDepthwiseConv1DLayerTest(test_utils.TestCase):

  def _GetRefParams(self, kernel_size, dim):
    builder = gshard_builder.MoEBuilder.Params().Set(
        model_dim=dim).Instantiate()
    return builder.DepthwiseConvAutoregressive('conv', kernel_size)

  def _GetParams(self, kernel_size, dim):
    p = gshard_layers.CausalDepthwiseConv1DLayer.Params().Set(
        name='conv',
        kernel_size=kernel_size,
        model_dims=dim,
        compatible_with_mtf_ckpt=True)
    return p

  def _GetInputs(self, batch, seqlen, dim):
    np.random.seed(None)
    return tf.convert_to_tensor(
        np.random.rand(batch, seqlen, dim).astype(np.float32))

  def testEqualToDepthwiseConvAutoregressive(self):
    b, seqlen, d, k = 2, 8, 4, 3

    with tf.variable_scope('ref'):
      ref_l = self._GetRefParams(k, d).Instantiate()
    with tf.variable_scope('act'):
      exp_l = self._GetParams(k, d).Instantiate()

    inputs = self._GetInputs(b, seqlen, d)
    # [b, t, d]
    ref_out = ref_l.FProp(ref_l.theta, inputs)
    # [b, t, d]
    act_out = exp_l.FProp(exp_l.theta, inputs)

    init_op = tf.global_variables_initializer()

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([ref_out, act_out])
      self.assertAllClose(expected, actual)


class Conv1DStateLayerTest(test_utils.TestCase):

  def _GetParams(self, kernel_size, dim):
    p = gshard_layers.CausalDepthwiseConv1DLayer.Params().Set(
        name='conv', kernel_size=kernel_size, model_dims=dim)
    p.state_layer = gshard_layers.Conv1DStateLayer.Params().Set(
        shape=[None, None, dim])
    return p

  def _GetInputs(self, batch, seqlen, dim):
    np.random.seed(None)
    np_inputs = np.random.rand(batch, seqlen, dim).astype(np.float32)
    tf.logging.info(f'np_inputs: {np_inputs}')
    return tf.convert_to_tensor(np_inputs)

  def testSingleStep(self):
    b, seqlen, dim, k, beam = 2, 8, 2, 3, 1

    inputs = self._GetInputs(b, seqlen * beam, dim)

    l = self._GetParams(k, dim).Instantiate()
    # Normal Fprop with a len=seqlen sequence.
    outputs = l.FProp(l.theta, inputs)

    state0 = gshard_layers.StateLayer.InitState(l, [b, beam, k])
    tf.logging.info(f'state0: {repr(state0)}')

    all_outputs = []
    state_t = state0
    theta_t = l.theta.DeepCopy()
    for i in range(seqlen):
      inputs_t = inputs[:, i:i + 1 * beam, :]

      # Copies state to theta.
      theta_t = gshard_layers.StateLayer.UpdateTheta(l, theta_t, state_t, t=i)
      tf.logging.info(f'theta_{i}: {repr(theta_t)}')

      # Updates theta inplace.
      out_t = l.FProp(theta_t, inputs_t)

      # Copies theta to state.
      state_t = gshard_layers.StateLayer.UpdateState(l, theta_t, state_t)
      tf.logging.info(f'state_{i}: {repr(state_t)}')

      all_outputs.append(out_t)

    # seqlen steps of FProp(), each with len=1.
    concat_step_outputs = tf.concat(all_outputs, axis=1)

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([outputs, concat_step_outputs])
      print(f'expected: {expected}')
      print(f'actual: {actual}')
      self.assertAllClose(expected, actual)


if __name__ == '__main__':
  tf.test.main()
