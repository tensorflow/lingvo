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

  def _GetInputs(self, batch, seq_len, dim):
    np.random.seed(None)
    return tf.convert_to_tensor(
        np.random.rand(batch, seq_len, dim).astype(np.float32))

  def testEqualToDepthwiseConvAutoregressive(self):
    b, seq_len, d, k = 2, 8, 4, 3

    with tf.variable_scope('ref'):
      ref_l = self._GetRefParams(k, d).Instantiate()
    with tf.variable_scope('act'):
      exp_l = self._GetParams(k, d).Instantiate()

    inputs = self._GetInputs(b, seq_len, d)
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

  def _GetParams(self,
                 kernel_size,
                 dims,
                 state_layer=True,
                 skip_store_zero_state=False):
    p = gshard_layers.CausalDepthwiseConv1DLayer.Params().Set(
        name='conv', kernel_size=kernel_size, model_dims=dims)
    if state_layer:
      p.state_layer = gshard_layers.Conv1DStateLayer.Params().Set(
          shape=[None, None] + dims,
          skip_store_zero_state=skip_store_zero_state)
    return p

  def _GetInputs(self, batch, seq_len, dim):
    np.random.seed(None)
    np_inputs = np.random.rand(batch, seq_len, dim).astype(np.float32)
    tf.logging.info(f'np_inputs: {np_inputs}')
    return tf.convert_to_tensor(np_inputs)

  def testSingleStep(self):
    b, seq_len, dim, k, beam = 2, 8, 2, 3, 1

    inputs = self._GetInputs(b, seq_len * beam, dim)
    # Normal Fprop with a len=seqlen sequence.
    l = self._GetParams(k, [dim]).Instantiate()
    outputs = l.FProp(l.theta, inputs)

    state0 = gshard_layers.StateLayer.InitState(l, [b, beam, k])
    tf.logging.info(f'state0: {repr(state0)}')

    all_outputs = []
    state_t = state0
    theta_t = l.theta.DeepCopy()
    for i in range(seq_len):
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

    # seq_len steps of FProp(), each with len=1.
    concat_step_outputs = tf.concat(all_outputs, axis=1)

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([outputs, concat_step_outputs])
      print(f'expected: {expected}')
      print(f'actual: {actual}')
      self.assertAllClose(expected, actual)

  def testSingleStepRank4(self):
    b, seq_len, dim1, dim2, k, beam = 2, 8, 2, 7, 3, 1

    inputs = self._GetInputs(b, seq_len * beam, dim1 * dim2)
    inputs = tf.reshape(inputs, (b, seq_len * beam, dim1, dim2))

    l = self._GetParams(k, [dim1, dim2]).Instantiate()
    # Normal Fprop with a len=seq_len sequence.
    outputs = l.FProp(l.theta, inputs)

    state0 = gshard_layers.StateLayer.InitState(l, [b, beam, k])
    tf.logging.info(f'state0: {repr(state0)}')

    all_outputs = []
    state_t = state0
    theta_t = l.theta.DeepCopy()
    for i in range(seq_len):
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

    # seq_len steps of FProp(), each with len=1.
    concat_step_outputs = tf.concat(all_outputs, axis=1)

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      expected, actual = sess.run([outputs, concat_step_outputs])
      print(f'expected: {expected}')
      print(f'actual: {actual}')
      self.assertAllClose(expected, actual)

  def testPrefix(self):
    b, prefix_len, seq_len, dim1, dim2, k, beam = 2, 5, 15, 2, 7, 3, 4

    inputs = self._GetInputs(b, seq_len * beam, dim1 * dim2)
    inputs = tf.reshape(inputs, (b, seq_len * beam, dim1, dim2))
    prefix = self._GetInputs(b, prefix_len, dim1 * dim2)
    prefix = tf.reshape(prefix, (b, prefix_len, dim1, dim2))
    prefix_and_inputs = tf.reshape(prefix, (b, 1, prefix_len, dim1, dim2))
    prefix_and_inputs = tf.tile(prefix_and_inputs, (1, beam, 1, 1, 1))
    prefix_and_inputs = tf.concat(
        [prefix_and_inputs,
         tf.reshape(inputs, (b, beam, seq_len, dim1, dim2))],
        axis=2)
    prefix_and_inputs = tf.reshape(prefix_and_inputs,
                                   (b * beam,
                                    (prefix_len + seq_len), dim1, dim2))

    with tf.variable_scope('model'):
      l_no_prefix = self._GetParams(
          k, [dim1, dim2], state_layer=False).Instantiate()
    with tf.variable_scope('model', reuse=True):
      l = self._GetParams(k, [dim1, dim2]).Instantiate()
    prefix_expected_outputs = l_no_prefix.FProp(l.theta, prefix)
    decode_expected_outputs = tf.reshape(
        l_no_prefix.FProp(l.theta, prefix_and_inputs)[:, prefix_len:],
        (b, beam, seq_len, dim1, dim2))

    state0 = gshard_layers.StateLayer.InitState(l, [b, beam, k])
    tf.logging.info(f'state0: {repr(state0)}')

    state_prefix = state0
    theta_prefix = l.theta.DeepCopy()
    theta_prefix = gshard_layers.StateLayer.UpdateTheta(
        l, theta_prefix, state_prefix, t=0)
    tf.logging.info(f'theta_{0}: {repr(theta_prefix)}')
    prefix_actual_outputs = l.FProp(theta_prefix, prefix)
    state_prefix = gshard_layers.StateLayer.UpdateState(l, theta_prefix,
                                                        state_prefix)
    tf.logging.info(f'state_{0}: {repr(state_prefix)}')

    decode_outputs = []
    state_t = state0
    theta_t = l.theta.DeepCopy()
    for i in range(seq_len):
      inputs_t = tf.reshape(inputs, (b, beam, seq_len, dim1, dim2))[:, :, i]

      # Copies state to theta.
      theta_t = gshard_layers.StateLayer.UpdateTheta(l, theta_t, state_t, t=i)
      tf.logging.info(f'theta_{i}: {repr(theta_t)}')

      # Updates theta inplace.
      out_t = l.FProp(theta_t, inputs_t)

      # Copies theta to state.
      state_t = gshard_layers.StateLayer.UpdateState(l, theta_t, state_t)
      tf.logging.info(f'state_{i}: {repr(state_t)}')

      decode_outputs.append(tf.expand_dims(out_t, axis=2))

    # seq_len steps of FProp(), each with len=1.
    decode_actual_outputs = tf.concat(decode_outputs, axis=2)

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      prefix_expected, prefix_actual = sess.run(
          [prefix_expected_outputs, prefix_actual_outputs])
      self.assertAllClose(prefix_expected, prefix_actual)
      decode_expected, decode_actual = sess.run(
          [decode_expected_outputs, decode_actual_outputs])
      self.assertAllClose(decode_expected, decode_actual)

  def testPrefixSuffix(self):
    b, prefix_len, suffix_len, seq_len, dim1, dim2, k, beam = 2, 5, 5, 15, 2, 7, 3, 4

    inputs = self._GetInputs(b, seq_len * beam, dim1 * dim2)
    inputs = tf.reshape(inputs, (b, seq_len * beam, dim1, dim2))

    prefix = self._GetInputs(b, prefix_len, dim1 * dim2)
    prefix = tf.reshape(prefix, (b, prefix_len, dim1, dim2))
    prefix_tiled = tf.reshape(prefix, (b, 1, prefix_len, dim1, dim2))
    prefix_tiled = tf.tile(prefix_tiled, (1, beam, 1, 1, 1))

    suffix = self._GetInputs(b, beam * suffix_len, dim1 * dim2)
    suffix = tf.reshape(suffix, (b, beam, suffix_len, dim1, dim2))

    full_inputs = tf.concat([
        prefix_tiled,
        tf.reshape(inputs, (b, beam, seq_len, dim1, dim2)), suffix
    ],
                            axis=2)
    full_inputs = tf.reshape(full_inputs,
                             (b * beam,
                              (prefix_len + seq_len + suffix_len), dim1, dim2))

    with tf.variable_scope('model'):
      l_no_prefix = self._GetParams(
          k, [dim1, dim2], state_layer=False).Instantiate()
    with tf.variable_scope('model', reuse=True):
      l = self._GetParams(k, [dim1, dim2]).Instantiate()
    prefix_expected_outputs = l_no_prefix.FProp(l.theta, prefix)
    decode_raw_outputs = l_no_prefix.FProp(l.theta, full_inputs)
    decode_expected_outputs = tf.reshape(
        decode_raw_outputs[:, prefix_len:-suffix_len],
        (b, beam, seq_len, dim1, dim2))
    suffix_expected_outputs = tf.reshape(decode_raw_outputs[:, -suffix_len:],
                                         (b, beam, suffix_len, dim1, dim2))

    state0 = gshard_layers.StateLayer.InitState(l, [b, beam, k])
    tf.logging.info(f'state0: {repr(state0)}')

    state_prefix = state0
    theta_prefix = l.theta.DeepCopy()
    theta_prefix = gshard_layers.StateLayer.UpdateTheta(
        l, theta_prefix, state_prefix, t=0)
    tf.logging.info(f'theta_{0}: {repr(theta_prefix)}')
    prefix_actual_outputs = l.FProp(theta_prefix, prefix)
    state_prefix = gshard_layers.StateLayer.UpdateState(l, theta_prefix,
                                                        state_prefix)
    tf.logging.info(f'state_{0}: {repr(state_prefix)}')

    decode_outputs = []
    state_t = state0
    theta_t = l.theta.DeepCopy()
    for i in range(seq_len):
      inputs_t = tf.reshape(inputs, (b, beam, seq_len, dim1, dim2))[:, :, i]

      # Copies state to theta.
      theta_t = gshard_layers.StateLayer.UpdateTheta(l, theta_t, state_t, t=i)
      tf.logging.info(f'theta_{i}: {repr(theta_t)}')

      # Updates theta inplace.
      out_t = l.FProp(theta_t, inputs_t)

      # Copies theta to state.
      state_t = gshard_layers.StateLayer.UpdateState(l, theta_t, state_t)
      tf.logging.info(f'state_{i}: {repr(state_t)}')

      decode_outputs.append(tf.expand_dims(out_t, axis=2))

    # seq_len steps of FProp(), each with len=1.
    decode_actual_outputs = tf.concat(decode_outputs, axis=2)

    theta_t = gshard_layers.StateLayer.UpdateTheta(
        l, theta_t, state_t, t=seq_len)
    suffix = tf.transpose(suffix, [0, 2, 1, 3, 4])
    suffix = tf.reshape(suffix, (b, suffix_len * beam, dim1, dim2))
    suffix_actual_outputs = l.FProp(theta_t, suffix)
    suffix_actual_outputs = tf.reshape(suffix_actual_outputs,
                                       (b, suffix_len, beam, dim1, dim2))
    suffix_actual_outputs = tf.transpose(suffix_actual_outputs, [0, 2, 1, 3, 4])

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      prefix_expected, prefix_actual = sess.run(
          [prefix_expected_outputs, prefix_actual_outputs])
      self.assertAllClose(prefix_expected, prefix_actual)
      decode_expected, decode_actual = sess.run(
          [decode_expected_outputs, decode_actual_outputs])
      self.assertAllClose(decode_expected, decode_actual)
      suffix_expected, suffix_actual = sess.run(
          [suffix_expected_outputs, suffix_actual_outputs])
      self.assertAllClose(suffix_expected, suffix_actual)

  # Test suffix logic when there is a gap between the decoding portion and the
  # suffix.
  def testPrefixSuffixGap(self):
    b, prefix_len, suffix_len, seq_len, dim1, dim2, k, beam = 2, 5, 5, 15, 2, 7, 3, 4

    inputs = self._GetInputs(b, seq_len * beam, dim1 * dim2)
    inputs = tf.reshape(inputs, (b, seq_len * beam, dim1, dim2))

    prefix = self._GetInputs(b, prefix_len, dim1 * dim2)
    prefix = tf.reshape(prefix, (b, prefix_len, dim1, dim2))
    prefix_tiled = tf.reshape(prefix, (b, 1, prefix_len, dim1, dim2))
    prefix_tiled = tf.tile(prefix_tiled, (1, beam, 1, 1, 1))

    suffix = self._GetInputs(b, beam * suffix_len, dim1 * dim2)
    suffix = tf.reshape(suffix, (b, beam, suffix_len, dim1, dim2))

    full_inputs = tf.concat([
        prefix_tiled,
        tf.reshape(inputs, (b, beam, seq_len, dim1, dim2)), suffix
    ],
                            axis=2)
    full_inputs = tf.reshape(full_inputs,
                             (b * beam,
                              (prefix_len + seq_len + suffix_len), dim1, dim2))

    with tf.variable_scope('model'):
      l_no_prefix = self._GetParams(
          k, [dim1, dim2], state_layer=False).Instantiate()
    with tf.variable_scope('model', reuse=True):
      l = self._GetParams(
          k, [dim1, dim2], skip_store_zero_state=True).Instantiate()
    prefix_expected_outputs = l_no_prefix.FProp(l.theta, prefix)
    decode_raw_outputs = l_no_prefix.FProp(l.theta, full_inputs)
    decode_expected_outputs = tf.reshape(
        decode_raw_outputs[:, prefix_len:-suffix_len],
        (b, beam, seq_len, dim1, dim2))
    suffix_expected_outputs = tf.reshape(decode_raw_outputs[:, -suffix_len:],
                                         (b, beam, suffix_len, dim1, dim2))

    state0 = gshard_layers.StateLayer.InitState(l, [b, beam, k])
    tf.logging.info(f'state0: {repr(state0)}')

    state_prefix = state0
    theta_prefix = l.theta.DeepCopy()
    theta_prefix = gshard_layers.StateLayer.UpdateTheta(
        l, theta_prefix, state_prefix, t=0)
    tf.logging.info(f'theta_{0}: {repr(theta_prefix)}')
    prefix_actual_outputs = l.FProp(theta_prefix, prefix)
    state_prefix = gshard_layers.StateLayer.UpdateState(l, theta_prefix,
                                                        state_prefix)
    tf.logging.info(f'state_{0}: {repr(state_prefix)}')

    decode_outputs = []
    state_t = state0
    theta_t = l.theta.DeepCopy()
    inputs = tf.reshape(inputs, (b, beam, seq_len, dim1, dim2))
    inputs = tf.concat([inputs, tf.zeros_like(inputs)], axis=2)  # Add gap.
    for i in range(seq_len * 2):
      inputs_t = inputs[:, :, i]

      # Copies state to theta.
      theta_t = gshard_layers.StateLayer.UpdateTheta(l, theta_t, state_t, t=i)
      tf.logging.info(f'theta_{i}: {repr(theta_t)}')

      # Updates theta inplace.
      out_t = l.FProp(theta_t, inputs_t)

      # Copies theta to state.
      state_t = gshard_layers.StateLayer.UpdateState(l, theta_t, state_t)
      tf.logging.info(f'state_{i}: {repr(state_t)}')

      decode_outputs.append(tf.expand_dims(out_t, axis=2))

    # seq_len steps of FProp(), each with len=1.
    decode_actual_outputs = tf.concat(decode_outputs[:seq_len], axis=2)

    theta_t = gshard_layers.StateLayer.UpdateTheta(
        l, theta_t, state_t, t=seq_len)
    suffix = tf.transpose(suffix, [0, 2, 1, 3, 4])
    suffix = tf.reshape(suffix, (b, suffix_len * beam, dim1, dim2))
    suffix_actual_outputs = l.FProp(theta_t, suffix)
    suffix_actual_outputs = tf.reshape(suffix_actual_outputs,
                                       (b, suffix_len, beam, dim1, dim2))
    suffix_actual_outputs = tf.transpose(suffix_actual_outputs, [0, 2, 1, 3, 4])

    init_op = tf.global_variables_initializer()
    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      prefix_expected, prefix_actual = sess.run(
          [prefix_expected_outputs, prefix_actual_outputs])
      self.assertAllClose(prefix_expected, prefix_actual)
      decode_expected, decode_actual = sess.run(
          [decode_expected_outputs, decode_actual_outputs])
      self.assertAllClose(decode_expected, decode_actual)
      suffix_expected, suffix_actual = sess.run(
          [suffix_expected_outputs, suffix_actual_outputs])
      self.assertAllClose(suffix_expected, suffix_actual)

if __name__ == '__main__':
  test_utils.main()
