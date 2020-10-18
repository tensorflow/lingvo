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
from absl.testing import flagsaver
from absl.testing import parameterized

from lingvo import compat as tf
from lingvo.core import bn_layers
from lingvo.core import cluster_factory
from lingvo.core import conformer_layer
from lingvo.core import layers
from lingvo.core import py_utils
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

  @parameterized.named_parameters(
      ('Basic',),
      ('BasicGN', False, 'gn'),
      ('SkipNorm', True),
  )
  def testStreamStep(self, testonly_skip_norm_layers=False, norm_type='ln'):
    with flagsaver.flagsaver(testonly_skip_norm_layers=testonly_skip_norm_layers
                            ), cluster_factory.SetEval(True):
      assert norm_type in ('ln', 'gn')
      batch, max_seqlen, input_dim, kernel = 2, 8, 2, 3
      p = conformer_layer.LConvLayer.CommonParams(
          input_dim=input_dim, is_causal=True, kernel_size=kernel)
      if norm_type == 'ln':
        p.conv_norm_layer_tpl = layers.LayerNorm.Params()
      else:
        p.conv_norm_layer_tpl = bn_layers.GroupNormLayer.Params().Set(
            num_groups=2, cumulative=True)
      p.name = 'lconv'

      l = p.Instantiate()
      init_op = tf.global_variables_initializer()

      np.random.seed(None)
      inputs = np.random.normal(
          0.1, 0.5, [batch, max_seqlen, input_dim]).astype(np.float32)
      print(f'np.sum(inputs): {np.sum(inputs)}')
      inputs = tf.convert_to_tensor(inputs)

      seqlen = np.random.randint(
          low=1, high=max_seqlen + 1, size=(batch,), dtype=np.int32)
      print(repr(seqlen))
      seqlen = tf.convert_to_tensor(seqlen)
      paddings = py_utils.PaddingsFromLengths(seqlen, max_seqlen)
      base_outputs, _ = l.FProp(l.theta, inputs, paddings)
      base_outputs *= tf.expand_dims(1. - paddings, -1)

      outputs = []
      state = l.zero_state(batch)
      for i in range(max_seqlen):
        output, _, state = l.StreamStep(l.theta, inputs[:, i:(i + 1), :],
                                        paddings[:, i:(i + 1)], state)
        outputs.append(output)
      # [b, t, d]
      outputs = tf.concat(outputs, axis=1)
      outputs *= tf.expand_dims(1. - paddings, -1)

      with self.session(use_gpu=False) as sess:
        sess.run(init_op)
        expected, actual = sess.run([base_outputs, outputs])
        print(repr(expected))
        print(repr(actual))
        print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
        print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
        self.assertAllClose(expected, actual)


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

  @parameterized.named_parameters(
      ('Basic',),
      ('BasicGN', False, 'gn'),
      ('BasicGNG1', False, 'gn', 1),
      ('BasicGNG8', False, 'gn', 8),
      ('SkipNorm', True),
      ('SkipNormGN', True),
  )
  def testStreamStep(self,
                     testonly_skip_norm_layers=False,
                     norm_type='ln',
                     num_groups=2):
    assert norm_type in ('ln', 'gn'), norm_type
    with flagsaver.flagsaver(testonly_skip_norm_layers=testonly_skip_norm_layers
                            ), cluster_factory.SetEval(True):
      batch, max_seqlen, input_dim, kernel = 2, 16, 8, 3
      num_heads, left_context, ffn_dim = 2, 3, 4
      p = conformer_layer.ConformerLayer.CommonParams(
          input_dim=input_dim,
          is_causal=True,
          atten_num_heads=num_heads,
          atten_left_context=left_context,
          atten_right_context=0,
          use_relative_atten=False,
          fflayer_hidden_dim=ffn_dim,
          kernel_size=kernel,
          layer_order='conv_before_mhsa')
      if norm_type == 'ln':
        p.lconv_tpl.conv_norm_layer_tpl = layers.LayerNorm.Params()
      else:
        p.lconv_tpl.conv_norm_layer_tpl = bn_layers.GroupNormLayer.Params().Set(
            num_groups=num_groups, cumulative=True)
      p.name = 'conformer'

      l = p.Instantiate()
      init_op = tf.global_variables_initializer()

      np.random.seed(None)
      inputs = 5 * np.random.normal(
          0.1, 0.5, [batch, max_seqlen, input_dim]).astype(np.float32)
      print(f'np.sum(inputs): {np.sum(inputs)}')
      inputs = tf.convert_to_tensor(inputs)

      seqlen = np.random.randint(
          low=1, high=max_seqlen + 1, size=(batch,), dtype=np.int32)
      print(repr(seqlen))
      seqlen = tf.convert_to_tensor(seqlen)
      paddings = py_utils.PaddingsFromLengths(seqlen, max_seqlen)

      base_outputs, _ = l.FProp(l.theta, inputs, paddings)
      base_outputs *= tf.expand_dims(1. - paddings, -1)

      outputs = []
      state = l.zero_state(batch)
      for i in range(max_seqlen):
        output, _, state = l.StreamStep(l.theta, inputs[:, i:(i + 1), :],
                                        paddings[:, i:(i + 1)], state)
        outputs.append(output)
      # [b, t, d]
      outputs = tf.concat(outputs, axis=1)
      outputs *= tf.expand_dims(1. - paddings, -1)

      with self.session(use_gpu=False) as sess:
        sess.run(init_op)
        expected, actual = sess.run([base_outputs, outputs])
        print(repr(expected))
        print(repr(actual))
        print(f'np.sum(np.abs(expected)): {np.sum(np.abs(expected))}')
        print(f'np.sum(np.abs(actual)): {np.sum(np.abs(actual))}')
        tol = 1.e-6 if testonly_skip_norm_layers else 2.e-5
        self.assertAllClose(expected, actual, atol=tol, rtol=tol)


if __name__ == '__main__':
  tf.test.main()
