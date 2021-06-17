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
        name='conv', filter_shape=(kernel_size, dim, 1))
    return p

  def _GetInputs(self, batch, seqlen, dim):
    np.random.seed(2021)
    return tf.convert_to_tensor(
        np.random.rand(batch, seqlen, dim).astype(np.float32))

  def testEqualToDepthwiseConvAutoregressive(self):
    b, seqlen, d, k = 2, 8, 4, 3

    ref_l = self._GetRefParams(k, d).Instantiate()
    exp_l = self._GetParams(k, d).Instantiate()

    inputs = self._GetInputs(b, seqlen, d)
    # [b, t, d]
    ref_out = ref_l.FProp(ref_l.theta, inputs)
    # [b, t, d]
    act_out = exp_l.FProp(exp_l.theta, inputs)

    # k vars, each shape is [d].
    all_ref_ws = [getattr(ref_l, f'w_{i}').vars.scale for i in range(k)]
    # [k, d]
    ref_w = tf.stack(all_ref_ws, axis=0)
    print(f'ref_w: {ref_w}')
    # [k, 1, d, 1]
    exp_w = exp_l.vars.w

    init_op = tf.global_variables_initializer()
    copy_op = tf.assign(exp_w, ref_w[:, None, :, None])

    with self.session(use_gpu=False) as sess:
      sess.run(init_op)
      sess.run(copy_op)
      expected, actual = sess.run([ref_out, act_out])
      self.assertAllClose(expected, actual)


if __name__ == '__main__':
  tf.test.main()
