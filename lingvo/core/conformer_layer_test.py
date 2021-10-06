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
from lingvo.core import batch_major_attention
from lingvo.core import bn_layers
from lingvo.core import cluster_factory
from lingvo.core import conformer_layer
from lingvo.core import conv_layers_with_time_padding
from lingvo.core import gshard_builder
from lingvo.core import layers as lingvo_layers
from lingvo.core import py_utils
from lingvo.core import stream_step_test_base
from lingvo.core import test_utils

import numpy as np


class LConvLayerTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('BN',),
      ('GN', 'gn'),
  )
  def testBasic(self, norm='bn'):
    batch_size, seqlen, dim = 2, 16, 4
    inputs = tf.zeros([batch_size, seqlen, dim])
    paddings = tf.zeros([batch_size, seqlen])

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


class LConvLayerStreamStepTest(stream_step_test_base.StreamStepTestBase):

  def _GetParams(self, **kwargs):
    input_dim = kwargs['input_dim']
    kernel = kwargs['kernel']
    norm_type = kwargs['norm_type']

    p = conformer_layer.LConvLayer.CommonParams(
        input_dim=input_dim, is_causal=True, kernel_size=kernel)
    if norm_type == 'ln':
      p.conv_norm_layer_tpl = lingvo_layers.LayerNorm.Params()
    else:
      p.conv_norm_layer_tpl = bn_layers.GroupNormLayer.Params().Set(
          num_groups=2, cumulative=True)
    p.name = 'lconv'
    return p

  def _FProp(self, layer, inputs, paddings):
    return layer.FProp(layer.theta, inputs, paddings)

  def _GetFPropOutput(self, fprop_out):
    return fprop_out[0]

  @parameterized.named_parameters(
      ('Basic',),
      ('BasicGN', False, 'gn'),
      ('SkipNorm', True),
  )
  def testLeftContext(self, testonly_skip_norm_layers=False, norm_type='ln'):
    with flagsaver.flagsaver(testonly_skip_norm_layers=testonly_skip_norm_layers
                            ), cluster_factory.SetEval(True):
      assert norm_type in ('ln', 'gn')
      input_dim, kernel = 2, 3
      self._TestStreamStepHelper(
          num_heads=2, input_dim=input_dim, kernel=kernel, norm_type=norm_type)


class ConformerLayerTest(test_utils.TestCase, parameterized.TestCase):

  def __init__(self, *args):
    super().__init__(*args)
    self.batch_size = 2
    self.maxlen = 32
    self.dim = 4
    self.heads = 2
    self.context = 2

  def _GetCommonParamsKwargs(self):
    return dict(
        input_dim=self.dim,
        atten_num_heads=self.heads,
        atten_left_context=self.context + 1,
        atten_right_context=self.context,
        kernel_size=3,
        fflayer_hidden_dim=4 * self.dim)

  def _GetParams(self, **custom_kwargs):
    kwargs = self._GetCommonParamsKwargs()
    kwargs.update(custom_kwargs)
    p = conformer_layer.ConformerLayer.CommonParams(**kwargs)
    p.name = 'conformer_layer'
    return p

  def _GetInputs(self, dtype=tf.float32):
    inputs = np.random.rand(self.batch_size, self.maxlen,
                            self.dim).astype(np.float32)
    paddings = np.zeros((self.batch_size, self.maxlen), np.float32)

    seqlen = np.random.randint(0, self.maxlen, size=(self.batch_size,))
    for i in range(self.batch_size):
      for j in range(self.maxlen):
        paddings[i][j] = 1. if j >= seqlen[i] else 0.
    return tf.constant(inputs, dtype=dtype), tf.constant(paddings, dtype=dtype)

  def _GetGrad(self, l, inputs, paddings):
    in_nmap = py_utils.NestedMap(features=inputs, paddings=paddings)
    out_nmap = l.FPropDefaultTheta(in_nmap)
    loss = tf.reduce_sum(out_nmap.features)
    grads = tf.gradients(
        loss,
        l.vars.Flatten(),
        unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return out_nmap.features, grads

  @parameterized.named_parameters(
      ('Base',),
      ('Reordered', 'conv_before_mhsa'),
      ('NoLConv', 'mhsa', False),
      ('NoMhsa', 'conv', True),
      ('NoFFStart', 'mhsa_before_conv', True, False),
      ('Transformer', 'mhsa', False, False),
  )
  def testBasic(self,
                layer_order='mhsa_before_conv',
                has_lconv=True,
                has_fflayer_start=True):
    p = self._GetParams()
    p.layer_order = layer_order
    if not has_lconv:
      p.lconv_tpl = None
    if not has_fflayer_start:
      p.fflayer_start_tpl = None

    l = p.Instantiate()
    inputs, paddings = self._GetInputs()
    outputs, grads = self._GetGrad(l, inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      grad_vals = sess.run(grads)
      print([x.shape for x in out_vals])
      print([g.shape for g in grad_vals])

  @parameterized.named_parameters(
      ('F32FPropF32Input', tf.float32, tf.float32),
      ('F32FPropBF16Input', tf.float32, tf.bfloat16),
      ('BF16FPropF32Input', tf.bfloat16, tf.float32),
      ('BF16FPropBF16Input', tf.bfloat16, tf.bfloat16),
  )
  def testFPropDtypes(self, fprop_dtype, input_dtype):
    p = self._GetParams()
    # batch_norm does not support bfloat16 on CPU.
    p.lconv_tpl.conv_norm_layer_tpl = (
        bn_layers.GroupNormLayer.Params().Set(num_groups=2))
    p.cls.SetFPropDtype(p, fprop_dtype)

    l = p.Instantiate()
    inputs, paddings = self._GetInputs(dtype=input_dtype)
    outputs, grads = self._GetGrad(l, inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      grad_vals = sess.run(grads)
      print([x.shape for x in out_vals])
      print([g.shape for g in grad_vals])

  @parameterized.named_parameters(
      ('DefaultFp32', [], tf.float32),
      ('RegexDtypeFp16', [(r'.*(fflayer_[01]|linear_start|post)/w$', tf.float16)
                         ], tf.float16),
  )
  def testFPropDtypesWithListRegexDtypes(self, regex_dtypes, target_dtype):
    p = self._GetParams()
    p.lconv_tpl.conv_norm_layer_tpl = (
        bn_layers.GroupNormLayer.Params().Set(num_groups=2))
    p.list_regex_dtypes = regex_dtypes

    l = p.Instantiate()
    inputs, paddings = self._GetInputs()
    outputs, grads = self._GetGrad(l, inputs, paddings)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      grad_vals = sess.run(grads)
      print([x.shape for x in out_vals])
      print([g.shape for g in grad_vals])

    tf.assert_type(l.vars.fflayer_start.fflayer.fc[0].w, target_dtype)
    tf.assert_type(l.vars.fflayer_start.fflayer.fc[1].w, target_dtype)
    tf.assert_type(l.vars.fflayer_end.fflayer.fc[0].w, target_dtype)
    tf.assert_type(l.vars.fflayer_end.fflayer.fc[1].w, target_dtype)
    tf.assert_type(l.vars.lconv.linear_start.w, target_dtype)
    tf.assert_type(l.vars.trans_atten.atten.post.w, target_dtype)

  @parameterized.named_parameters(
      ('Start', True, False),
      ('End', False, True),
      ('StartAndEnd', True, True),
      ('None', False, False),
  )
  def testMoEFFLayerClassMethodInitParity(self, use_fflayer_start_moe,
                                          use_fflayer_end_moe):
    """Tests Conformer-MoE initializations via classmethods and explicitly."""

    num_experts, num_groups, num_devices, per_expert_capacity_dim = 2, 2, 2, 2
    # Create params setting MoEBuilder params explicitly.
    ref_kwargs = dict()
    if use_fflayer_start_moe:
      # Set MoEBuilder params explicitly.
      ref_kwargs['fflayer_start_tpl'] = gshard_builder.MoEBuilder.Params().Set(
          e_dim=num_experts,
          c_dim=per_expert_capacity_dim,
          num_devices=num_devices,
          num_groups=num_groups)
    if use_fflayer_end_moe:
      ref_kwargs['fflayer_end_tpl'] = gshard_builder.MoEBuilder.Params().Set(
          e_dim=num_experts,
          c_dim=per_expert_capacity_dim,
          num_devices=num_devices,
          num_groups=num_groups)
    ref_p = self._GetParams(**ref_kwargs)

    # Params setting MoEBuilder params via classmethod.
    moe_kwargs = dict()
    if use_fflayer_start_moe:
      # Set MoEBuilder params via classmethod.
      moe_kwargs['fflayer_start_tpl'] = conformer_layer.GShardMoELayerParams(
          num_devices, num_groups, num_experts, per_expert_capacity_dim)
    if use_fflayer_end_moe:
      moe_kwargs['fflayer_end_tpl'] = conformer_layer.GShardMoELayerParams(
          num_devices, num_groups, num_experts, per_expert_capacity_dim)
    moe_p = self._GetParams(**moe_kwargs)
    # Verify layer params are equal in both cases.
    with self.subTest('testParamsParity'):
      self.assertCountEqual(ref_p.ToText().split('\n'),
                            moe_p.ToText().split('\n'))

    # Test both initializations and verify moe sublayer.
    with self.subTest('testInit'):
      ref_p.name = 'ref_moe_conformer_layer'
      ref_layer = ref_p.Instantiate()
      moe_p.name = 'classmethod_moe_conformer_layer'
      moe_layer = moe_p.Instantiate()
      for layer in (ref_layer, moe_layer):
        if use_fflayer_start_moe:
          self.assertNotIn('fflayer_start', layer.children)
          self.assertIn('fflayer_start_moe', layer.children)
        if use_fflayer_end_moe:
          self.assertNotIn('fflayer_end', layer.children)
          self.assertIn('fflayer_end_moe', layer.children)

  @parameterized.named_parameters(
      ('Start', True, False, 0.593693),
      ('End', False, True, 0.4582923),
      ('StartAndEnd', True, True, 1.0213419),
      ('None', False, False, 0.0),
  )
  def testMoEFFLayerFProp(self, use_fflayer_start_moe, use_fflayer_end_moe,
                          expected_aux_loss):
    kwargs = {}
    if use_fflayer_start_moe:
      kwargs['fflayer_start_tpl'] = gshard_builder.MoEBuilder.Params().Set(
          e_dim=2, c_dim=2, num_devices=2)
    if use_fflayer_end_moe:
      kwargs['fflayer_end_tpl'] = gshard_builder.MoEBuilder.Params().Set(
          e_dim=2, c_dim=2, num_devices=2)
    p = self._GetParams(**kwargs)
    l = p.Instantiate()
    inputs, paddings = self._GetInputs()
    inputs = tf.convert_to_tensor(inputs)
    paddings = tf.convert_to_tensor(paddings)
    in_nmap = py_utils.NestedMap(features=inputs, paddings=paddings)
    in_nmap.aux_loss = tf.convert_to_tensor(0., py_utils.FPropDtype(p))
    out_nmap = l.FPropDefaultTheta(in_nmap)
    self.assertIn('aux_loss', out_nmap)
    loss = tf.reduce_sum(out_nmap.features) + 0.01 * out_nmap.aux_loss
    grads = tf.gradients(
        loss,
        l.vars.Flatten(),
        unconnected_gradients=tf.UnconnectedGradients.ZERO)

    with self.session() as sess:
      tf.global_variables_initializer().run()
      out_vals = sess.run(out_nmap.features)
      grad_vals = sess.run(grads)
      self.assertEqual(out_nmap.aux_loss.shape, ())
      aux_loss = sess.run(out_nmap.aux_loss)
      self.assertAlmostEqual(expected_aux_loss, aux_loss, places=5)
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

  @parameterized.named_parameters(
      ('Basic',),
      ('NegativeLocalContext', -1),
      ('NegativeLeftContext', None, -1, None),
      ('NegativeRightContext', None, None, -1),
      ('NegativeContext1', -1, None, -1),
      ('NegativeContext2', None, -1, -1),
      ('NegativeContext3', -1, -1, None),
      ('NegativeContext4', -1, None, -1),
      ('NegativeContext5', -1, -1, -1),
      ('NegativeContext6', None, None, None),
  )
  def testAttenContextParams(self,
                             local_context=None,
                             left_context=None,
                             right_context=None):
    """Tests atten context cfg params."""
    inputs, paddings = self._GetInputs()
    base_p_kwargs = self._GetCommonParamsKwargs()
    base_p_kwargs['atten_local_context'] = None
    base_p_kwargs['atten_left_context'] = None
    base_p_kwargs['atten_right_context'] = None
    base_p = conformer_layer.ConformerLayer.CommonParams(**base_p_kwargs)
    base_p.name = 'base'
    base_p.layer_order = 'conv_before_mhsa'

    new_p_kwargs = self._GetCommonParamsKwargs()
    new_p_kwargs['atten_local_context'] = local_context
    new_p_kwargs['atten_left_context'] = left_context
    new_p_kwargs['atten_right_context'] = right_context
    new_p = conformer_layer.ConformerLayer.CommonParams(**new_p_kwargs)
    new_p.name = 'new'
    new_p.layer_order = 'conv_before_mhsa'

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

  def testCustomAttentionLayer(self):
    p = self._GetParams()
    # Use a custom atten_tpl.
    p.trans_atten_tpl.atten_tpl = (
        batch_major_attention.MultiHeadedFavorAttention.Params().Set(
            num_random_features=4))
    layer = p.Instantiate()
    self.assertIsInstance(layer.trans_atten.atten,
                          batch_major_attention.MultiHeadedFavorAttention)

  @parameterized.named_parameters(
      ('Basic', 8, 'SWISH', 0.5),
      ('BasicReLU', 16, 'RELU', 1.),
  )
  def testFFlayerParams(self,
                        fflayer_hidden_dim=None,
                        fflayer_activation=None,
                        fflayer_residual_weight=0.5):
    p = self._GetParams(
        fflayer_hidden_dim=fflayer_hidden_dim,
        fflayer_activation=fflayer_activation,
        fflayer_residual_weight=fflayer_residual_weight)
    layer = p.Instantiate()

    start_fflayer = layer.fflayer_start
    actual_start_hidden_dim = start_fflayer.params.hidden_dim
    actual_start_activation = start_fflayer.params.activation
    actual_start_residual_weight = start_fflayer.params.residual_weight
    end_fflayer = layer.fflayer_end
    actual_end_hidden_dim = end_fflayer.params.hidden_dim
    actual_end_activation = end_fflayer.params.activation
    actual_end_residual_weight = end_fflayer.params.residual_weight

    self.assertEqual(fflayer_hidden_dim, actual_start_hidden_dim)
    self.assertEqual(fflayer_activation, actual_start_activation)
    self.assertEqual(fflayer_residual_weight, actual_start_residual_weight)
    self.assertEqual(fflayer_hidden_dim, actual_end_hidden_dim)
    self.assertEqual(fflayer_activation, actual_end_activation)
    self.assertEqual(fflayer_residual_weight, actual_end_residual_weight)

  @parameterized.named_parameters(
      ('shared', True),
      ('not_shared', False),
  )
  def testFFlayerWeightSharing(self, fflayer_weight_sharing):
    p = self._GetParams()
    p.fflayer_weight_sharing = fflayer_weight_sharing
    layer = p.Instantiate()

    # FFLayer variables will all have same full name iif weights are shared.
    def _VarNamesDebugString(vars_):
      return py_utils.Transform(lambda x: x.name, vars_).DebugString()

    fflayer_start_var_names = _VarNamesDebugString(layer.fflayer_start.vars)
    fflayer_end_var_names = _VarNamesDebugString(layer.fflayer_end.vars)

    self.assertEqual(fflayer_weight_sharing,
                     (fflayer_start_var_names == fflayer_end_var_names))

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
      ('WithoutRelPosAtten', False),
      ('WithRelPosAtten', True),
  )
  def testApplyGShard(self, use_relative_atten):
    with self.session() as sess:
      conformer_p = conformer_layer.ConformerLayer.CommonParams(
          input_dim=self.dim,
          atten_num_heads=self.heads,
          atten_local_context=self.context,
          use_relative_atten=use_relative_atten,
          kernel_size=2,
          fflayer_hidden_dim=4 * self.dim)
      conformer_p.name = 'conformer_layer'
      conformer_layer.ApplyGshard(
          conformer_p,
          device_mesh=[1, 2],
          proj_w_split_list=[[0, 1], [1, 0]],
          proj_activation_split_list=[[0, -1, 1], [0, -1, -1]],
          atten_dnh_w_split=[0, 1, -1],
          atten_blnh_activation_split=[0, -1, 1, -1],
          atten_bld_activation_split=[0, -1, -1],
          lconv_df_w_split=[0, 1],
          lconv_hwim_w_split=[-1, -1, 1, -1],
          lconv_fd_w_split=[-1, -1],
          lconv_blf_activation_split=[0, -1, 1],
          lconv_bld_activation_split=[0, -1, -1])
      inputs, paddings = self._GetInputs()
      conformer_l = conformer_p.Instantiate()
      outputs = conformer_l.FProp(
          conformer_l.theta,
          py_utils.NestedMap(
              features=tf.convert_to_tensor(inputs),
              paddings=tf.convert_to_tensor(paddings)))
      tf.logging.info('outputs=%s', outputs)
      tf.global_variables_initializer().run()
      out_vals = sess.run(outputs)
      print([x.shape for x in out_vals.Flatten()])

  @parameterized.named_parameters(
      ('Dropout', 'dropout_prob', 0.1),
      ('LayerOrder', 'layer_order', 'conv_before_mhsa'),
      ('FFLayerActivation', 'fflayer_activation', 'GELU'),
      ('UseRelativeAttentionTrue', 'use_relative_atten', True),
      ('UseRelativeAttentionFalse', 'use_relative_atten', False),
      ('IsCausal', 'is_causal', True),
      ('ListRegexDtypes', 'list_regex_dtypes', [('test_regex', tf.float16)]))
  def testCommonParamsSet(self, param_name, param_val):
    """Checks values set in CommonParams() correctly."""

    def _GetMinimalCommonParamsKwargs():
      """These args are required to be set to call CommonParams."""
      return dict(
          input_dim=2, atten_num_heads=4, kernel_size=3, fflayer_hidden_dim=8)

    kwargs = _GetMinimalCommonParamsKwargs()
    kwargs.update({param_name: param_val})
    if param_name == 'is_causal' and param_val:
      kwargs['atten_right_context'] = 0
      kwargs['use_relative_atten'] = False
    p = conformer_layer.ConformerLayer.CommonParams(**kwargs)
    p.name = 'conformer_layer'
    if param_name == 'use_relative_atten':
      atten_cls = p.trans_atten_tpl.atten_tpl.cls
      if param_val:
        self.assertTrue(
            issubclass(atten_cls, batch_major_attention.MultiHeadedAttentionXL),
            msg=atten_cls)
      else:
        self.assertTrue(
            issubclass(atten_cls, batch_major_attention.MultiHeadedAttention),
            msg=atten_cls)
    elif param_name == 'fflayer_activation':
      self.assertEqual(p.fflayer_start_tpl.activation, param_val)
      self.assertEqual(p.fflayer_end_tpl.activation, param_val)
    else:
      self.assertEqual(p.Get(param_name), param_val)


class ConformerLayerStreamStepTest(stream_step_test_base.StreamStepTestBase):

  def _GetParams(self, **kwargs):
    input_dim = kwargs['input_dim']
    kernel = kwargs['kernel']
    layer_order = kwargs['layer_order']
    num_heads = kwargs['num_heads']
    left_context = kwargs['left_context']
    right_context = kwargs['right_context']
    ffn_dim = kwargs['ffn_dim']
    # optional params.
    norm_type = kwargs.get('norm_type', 'gn')
    has_lconv = kwargs.get('has_lconv', 'conv2d')
    has_fflayer_start = kwargs.get('has_fflayer_start', True)
    num_groups = kwargs.get('num_groups', 2)

    if layer_order == 'mhsa':
      kernel = None
    p = conformer_layer.ConformerLayer.CommonParams(
        input_dim=input_dim,
        is_causal=True,
        atten_num_heads=num_heads,
        atten_left_context=left_context,
        atten_right_context=right_context,
        use_relative_atten=False,
        fflayer_hidden_dim=ffn_dim,
        kernel_size=kernel,
        layer_order=layer_order)
    if norm_type == 'ln':
      p.lconv_tpl.conv_norm_layer_tpl = lingvo_layers.LayerNorm.Params()
    else:
      p.lconv_tpl.conv_norm_layer_tpl = bn_layers.GroupNormLayer.Params().Set(
          num_groups=num_groups, cumulative=True)
    if not has_lconv:
      p.lconv_tpl = None
    elif has_lconv == 'conv2d':
      p.lconv_tpl.depthwise_conv_tpl = (
          conv_layers_with_time_padding.CausalConv2DLayerWithPadding.Params())
    else:
      assert has_lconv == 'depthwise'
    if not has_fflayer_start:
      p.fflayer_start_tpl = None

    p.name = 'conformer'
    return p

  def _FProp(self, layer, inputs, paddings):
    return layer.FProp(layer.theta,
                       py_utils.NestedMap(features=inputs, paddings=paddings))

  def _GetFPropOutput(self, fprop_out):
    return fprop_out.features

  @parameterized.named_parameters(
      {
          'testcase_name': 'Basic',
      },
      {
          'testcase_name': 'BasicGN',
          'norm_type': 'gn'
      },
      {
          'testcase_name': 'BasicGN1',
          'norm_type': 'gn',
          'num_groups': 1
      },
      {
          'testcase_name': 'BasicGN8',
          'norm_type': 'gn',
          'num_groups': 8
      },
      {
          'testcase_name': 'SkipNorm',
          'testonly_skip_norm_layers': True
      },
      {
          'testcase_name': 'SkipNormGN',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn'
      },
      {
          'testcase_name': 'SkipNormGNR1',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn',
          'right_context': 1,
      },
      {
          'testcase_name': 'SkipNormGNR2',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn',
          'right_context': 2,
      },
      {
          'testcase_name': 'SkipNormGNStride2',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn',
          'stride': 2
      },
      {
          'testcase_name': 'SkipNormGNStride4',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn',
          'stride': 4
      },
      {
          'testcase_name': 'SkipNormGNStride2R1',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn',
          'stride': 2,
          'right_context': 1
      },
      {
          'testcase_name': 'SkipNormGNStride4R2',
          'testonly_skip_norm_layers': True,
          'norm_type': 'gn',
          'stride': 4,
          'right_context': 2
      },
      {
          'testcase_name': 'Reordered',
          'layer_order': 'mhsa_before_conv'
      },
      {
          'testcase_name': 'Conv2D',
          'has_lconv': 'conv2d',
      },
      {
          'testcase_name': 'NoLConv',
          'layer_order': 'mhsa',
          'has_lconv': False
      },
      {
          'testcase_name': 'NoMhsa',
          'layer_order': 'conv'
      },
      {
          'testcase_name': 'NoFFStart',
          'layer_order': 'conv_before_mhsa',
          'has_fflayer_start': False
      },
      {
          'testcase_name': 'Transformer',
          'layer_order': 'mhsa',
          'has_lconv': False,
          'has_fflayer_start': False
      },
      {
          'testcase_name': 'TransformerSkipNormR2',
          'testonly_skip_norm_layers': True,
          'layer_order': 'mhsa',
          'has_lconv': False,
          'has_fflayer_start': False,
          'right_context': 2,
      },
  )
  def testCommon(self,
                 testonly_skip_norm_layers=False,
                 norm_type='ln',
                 num_groups=2,
                 stride=1,
                 layer_order='conv_before_mhsa',
                 has_lconv='depthwise',
                 has_fflayer_start=True,
                 right_context=0):
    assert norm_type in ('ln', 'gn'), norm_type
    kwargs = dict(
        input_dim=8,
        kernel=3,
        layer_order=layer_order,
        num_heads=2,
        left_context=3,
        right_context=right_context,
        ffn_dim=4,
        stride=stride,
        norm_type=norm_type,
        has_lconv=has_lconv,
        has_fflayer_start=has_fflayer_start,
        num_groups=num_groups)
    kwargs['tol'] = 1e-5
    with cluster_factory.SetEval(True), flagsaver.flagsaver(
        testonly_skip_norm_layers=testonly_skip_norm_layers):
      self._TestStreamStepHelper(**kwargs)

  def testStackingLayerWithRightContext(self):
    tf.random.set_seed(2021)
    kwargs = dict(
        input_dim=8,
        kernel=3,
        num_heads=2,
        left_context=6,
        right_context=3,
        ffn_dim=4,
        stride=2,
        layer_order='mhsa_before_conv',
        num_layers=3)
    with cluster_factory.SetEval(True):
      self._TestRightContextStackingLayersHelper(**kwargs)


if __name__ == '__main__':
  tf.test.main()
