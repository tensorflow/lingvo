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
from unittest import mock
from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import bn_layers
from lingvo.core import conformer_layer
from lingvo.core import test_utils

import numpy as np


class LConvLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('BN',),
      ('GN', 'gn'),
  )
  def testBasic(self, norm='bn'):
    batch, seqlen, dim = 2, 16, 4
    inputs = tf.zeros([batch, seqlen, dim])
    paddings = tf.zeros([batch, seqlen])

    p = conformer_layer.LConvLayer.CommonParams(input_dim=dim, kernel_size=3)
    p.name = 'lconv_layer'
    if norm == 'gn':
      # default is bn
      p.conv_norm_layer_tpl = (
          bn_layers.GroupNormLayer.Params().Set(num_groups=2))
    elif norm != 'bn':
      raise ValueError('Only gn and bn are supported.')

    l = p.Instantiate()
    outputs = l.FPropDefaultTheta(inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      print([x.shape for x in out_vals])


class ConformerLayerTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch = 2
    self.maxlen = 32
    self.dim = 4
    self.heads = 2
    self.context = 2

  def _GetParams(self):
    p = conformer_layer.ConformerLayer.CommonParams(
        input_dim=self.dim,
        atten_num_heads=self.heads,
        atten_left_context=self.context + 1,
        atten_right_context=self.context,
        kernel_size=3,
        fflayer_hidden_dim=4 * self.dim)
    p.name = 'conformer_layer'
    return p

  def _GetInputs(self):
    inputs = np.random.rand(self.batch, self.maxlen,
                            self.dim).astype(np.float32)
    paddings = np.zeros((self.batch, self.maxlen), np.float32)

    seqlen = np.random.randint(0, self.maxlen, size=(self.batch,))
    for i in range(self.batch):
      for j in range(self.maxlen):
        paddings[i][j] = 1. if j >= seqlen[i] else 0.
    return inputs, paddings

  def _GetGrad(self, l, inputs, paddings):
    outputs, _ = l.FPropDefaultTheta(inputs, paddings)
    loss = tf.reduce_sum(outputs)
    grads = tf.gradients(
        loss,
        l.vars.Flatten(),
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return outputs, grads

  @parameterized.named_parameters(
      ('Base',),
      ('Reordered', 'conv_before_mhsa'),
  )
  def testBasic(self, layer_order='mhsa_before_conv'):

    p = self._GetParams()
    p.layer_order = layer_order

    l = p.Instantiate()
    inputs, paddings = self._GetInputs()
    outputs, grads = self._GetGrad(l, inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      grad_vals = sess.run(grads)
      print([x.shape for x in out_vals])
      print([g.shape for g in grad_vals])

  def testRemat(self):
    inputs, paddings = self._GetInputs()
    base_p = self._GetParams()
    base_p.name = 'base'
    base_p.layer_order = 'conv_before_mhsa'

    new_p = base_p.Copy()
    new_p.name = 'new'
    new_p.remat = True

    base_l = base_p.Instantiate()
    new_l = new_p.Instantiate()

    _, base_grads = self._GetGrad(base_l, inputs, paddings)
    base_grads = base_l.vars.Pack(base_grads)

    _, new_grads = self._GetGrad(new_l, inputs, paddings)
    new_grads = new_l.vars.Pack(new_grads)

    assign_op = [
        tf.assign(dst, src)
        for (src, dst) in zip(base_l.vars.Flatten(), new_l.vars.Flatten())
    ]
    init_op = tf.global_variables_initializer()
    with self.session() as sess:
      sess.run(init_op)
      sess.run(assign_op)
      base_grads_val = sess.run(base_grads)
      new_grads_val = sess.run(new_grads)

      for (k, v1), (_, v2) in zip(base_grads_val.FlattenItems(),
                                  new_grads_val.FlattenItems()):
        self.assertAllClose(v1, v2, msg=k)

  def testCommonParamsAbuse(self):
    """Checks CommonParams() is not called in __init__()."""
    p = self._GetParams()
    with mock.patch(
        'lingvo.core.conformer_layer.ConformerLayer.CommonParams',
        autospec=True) as m1:
      with mock.patch(
          'lingvo.core.conformer_layer.LConvLayer.CommonParams',
          autospec=True) as m2:
        p.Instantiate()
        self.assertFalse(m1.called)
        self.assertFalse(m2.called)


if __name__ == '__main__':
  tf.test.main()
