# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for conformer layers as in https://arxiv.org/abs/2005.08100."""
# Lint as: PY3
from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import conformer_layer
from lingvo.core import test_utils


class LConv2DLayerTest(test_utils.TestCase, parameterized.TestCase):

  def testBasic(self):
    batch, seqlen, dim = 2, 16, 4
    inputs = tf.zeros([batch, seqlen, dim])
    paddings = tf.zeros([batch, seqlen])

    p = conformer_layer.LConv2DLayer.CommonParams(input_dim=dim, kernel_size=3)
    p.name = 'lconv_layer'
    l = p.Instantiate()
    outputs = l.FPropDefaultTheta(inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      print([x.shape for x in out_vals])


class ConformerLayerTest(test_utils.TestCase, parameterized.TestCase):

  def testBasic(self):
    batch, seqlen, dim, heads = 2, 32, 4, 2
    context = 2

    inputs = tf.zeros([batch, seqlen, dim])
    paddings = tf.zeros([batch, seqlen])

    p = conformer_layer.ConformerLayer.CommonParams(
        input_dim=dim,
        atten_num_heads=heads,
        atten_left_context=context + 1,
        atten_right_context=context,
        kernel_size=3,
        fflayer_hidden_dim=4 * dim)
    p.name = 'conformer_layer'
    l = p.Instantiate()
    outputs = l.FPropDefaultTheta(inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      print([x.shape for x in out_vals])


if __name__ == '__main__':
  tf.test.main()
