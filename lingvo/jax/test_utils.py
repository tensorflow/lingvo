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
"""Utility functions for JAX tests."""

from typing import Any, Optional

from absl import flags
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS
JTensor = jnp.ndarray
NestedMap = py_utils.NestedMap


_dtype = lambda x: getattr(x, 'dtype', None) or np.asarray(x).dtype


class TestCase(parameterized.TestCase):
  """Test method for lingvo tests."""

  def assertAllClose(self,
                     x,
                     y,
                     check_dtypes=True,
                     rtol=1E-5,
                     atol=1E-5,
                     **kwargs):
    """Wrapper for np.testing.assert_allclose()."""
    x = np.asarray(x)
    y = np.asarray(y)
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    x = x.astype(np.float32) if x.dtype == jnp.bfloat16 else x
    y = y.astype(np.float32) if y.dtype == jnp.bfloat16 else y
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, **kwargs)

  def assertArraysEqual(self, x, y, check_dtypes=True, **kwargs):
    """Wrapper for np.testing.assert_array_equal()."""
    x = np.asarray(x)
    y = np.asarray(y)
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    np.testing.assert_array_equal(x, y, **kwargs)

  def assertDtypesMatch(self, x, y):
    self.assertEqual(
        jax.dtypes.canonicalize_dtype(_dtype(x)),
        jax.dtypes.canonicalize_dtype(_dtype(y)))


def to_np(x: JTensor) -> np.ndarray:
  """Converts TF/JAX tensors to numpy."""
  return np.asarray(x, dtype=np.float32)


def unshard_input_nmap(x_nmap: NestedMap) -> NestedMap:
  """Unshards input sequences.

  Args:
    x_nmap: NestedMap with some tensors of shape [num_devices, batch, ...].

  Returns:
    NestedMap with tensors reshaped to [num_devices * batch, seq_len].
  """

  def unshard(x: JTensor) -> JTensor:
    num_devices = x.shape[0]
    batch_size = x.shape[1]
    new_shape = [num_devices * batch_size] + x.shape[2:]  # pytype: disable=unsupported-operands  # jax-ndarray
    return tf.reshape(x, new_shape)

  return tf.nest.map_structure(unshard, x_nmap)


def to_tf_nmap(x_nmap: NestedMap) -> NestedMap:
  """Converts numpy dtypes to TF dtypes in a variable collection."""

  def to_tf(x: Any) -> JTensor:
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    if x.dtype == np.float32:
      return tf.constant(np.asarray(x, x.dtype), tf.float32)
    elif x.dtype == np.int32:
      return tf.constant(np.asarray(x, x.dtype), tf.int32)
    if x.dtype == np.float64:
      return tf.constant(np.asarray(x, x.dtype), tf.float64)
    elif x.dtype == np.int64:
      return tf.constant(np.asarray(x, x.dtype), tf.int64)
    elif x.dtype == np.uint32:
      return tf.constant(np.asarray(x, x.dtype), tf.uint32)
    else:
      assert 'dtype not supported yet'  # pytype: disable=bad-return-type  # jax-ndarray

  return tf.nest.map_structure(to_tf, x_nmap)


def apply(layer, layer_vars, method, *args, context_p=None, seed=123, **kwargs):
  prng_key = jax.random.PRNGKey(seed=seed)
  with base_layer.JaxContext.new_context(
      params=context_p,
      prng_key=prng_key,
      global_step=jnp.array(0, dtype=jnp.uint32)) as jax_context:
    jax_context.bind(layer, layer.vars_to_flax_vars(layer_vars))
    return method(*args, **kwargs)


def replace_jax_transformer_ffwd_vars_to_tf(
    jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces JAX TransformerFeedForward vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX TransformerFeedforward layer vars.

  Returns:
    tf_initial_vars which is TF compatible.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars = py_utils.NestedMap()
  tf_initial_vars.fflayer = py_utils.NestedMap(
      fc=[
          py_utils.NestedMap(
              w=jax_initial_vars.ffn_layer1.linear.w,
              b=jax_initial_vars.ffn_layer1.bias.b),
          py_utils.NestedMap(
              w=jax_initial_vars.ffn_layer2.linear.w,
              b=jax_initial_vars.ffn_layer2.bias.b),
      ],
      dropout=[py_utils.NestedMap(), py_utils.NestedMap()],
  )
  tf_initial_vars.layer_norm = py_utils.NestedMap(
      bias=jax_initial_vars.layer_norm.bias,
      scale=jax_initial_vars.layer_norm.scale)
  tf_initial_vars.residual_dropout = py_utils.NestedMap()
  tf_initial_vars.residual_droppath = py_utils.NestedMap()
  return tf_initial_vars


def replace_jax_attention_vars_to_tf(
    jax_initial_vars: NestedMap,
    cross_attention: Optional[bool] = False) -> NestedMap:
  """Replaces JAX attention vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX attention layer vars.
    cross_attention: Whether cross attention is involved.

  Returns:
    tf_initial_vars which is TF compatible.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.fflayer = jax_initial_vars.ff_layer
  tf_initial_vars.fflayer.fflayer = py_utils.NestedMap()
  is_moe = 'gate' in jax_initial_vars.ff_layer
  if is_moe:
    tf_initial_vars.fflayer.fflayer.layer_norm = py_utils.NestedMap()
    tf_initial_vars.fflayer.fflayer.layer_norm.scale = jax_initial_vars.ff_layer.layer_norm.scale
    tf_initial_vars.fflayer.fflayer.layer_norm.bias = jax_initial_vars.ff_layer.layer_norm.bias
    tf_initial_vars.fflayer.fflayer.gate = jax_initial_vars.ff_layer.gate
    tf_initial_vars.fflayer.fflayer.wi_0 = jax_initial_vars.ff_layer.wi_0
    tf_initial_vars.fflayer.fflayer.wo_0 = jax_initial_vars.ff_layer.wo_0
  else:
    tf_initial_vars.fflayer.fflayer.dropout = [1.0, 1.0]
    tf_initial_vars.fflayer.fflayer.fc = [NestedMap(), NestedMap()]
    tf_initial_vars.fflayer.fflayer.fc[0].w = (
        jax_initial_vars.ff_layer.ffn_layer1.linear.w)
    tf_initial_vars.fflayer.fflayer.fc[0].b = (
        jax_initial_vars.ff_layer.ffn_layer1.bias.b)
    tf_initial_vars.fflayer.fflayer.fc[1].w = (
        jax_initial_vars.ff_layer.ffn_layer2.linear.w)
    tf_initial_vars.fflayer.fflayer.fc[1].b = (
        jax_initial_vars.ff_layer.ffn_layer2.bias.b)
  tf_initial_vars.self_atten = NestedMap()
  tf_initial_vars.self_atten.layer_norm = jax_initial_vars.layer_norm
  tf_initial_vars.self_atten.atten = jax_initial_vars.self_attention
  tf_initial_vars.self_atten.residual_dropout = 1.0
  if cross_attention:
    tf_initial_vars.cross_atten = NestedMap()
    tf_initial_vars.cross_atten.layer_norm = jax_initial_vars.layer_norm
    tf_initial_vars.cross_atten.atten = jax_initial_vars.cross_attention
    tf_initial_vars.cross_atten.residual_dropout = 1.0
  return tf_initial_vars


def replace_jax_single_shard_full_softmax_vars_to_tf(
    jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces JAX Single Shard Full Softmax vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX ConvBNAct layer vars.

  Returns:
    tf_initial_vars which is TF compatible with ConvBNAct.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.linear = py_utils.NestedMap(
      w=jax_initial_vars.logits_ffn.linear.w)
  tf_initial_vars.bias = py_utils.NestedMap(
      b=jax_initial_vars.logits_ffn.bias.b)
  del tf_initial_vars.logits_ffn
  return tf_initial_vars


def replace_jax_simple_full_softmax_vars_to_tf(
    jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces JAX Simple Full Softmax vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX ConvBNAct layer vars.

  Returns:
    tf_initial_vars which is TF compatible with ConvBNAct.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.weight_0 = jax_initial_vars.logits_ffn.linear.w
  tf_initial_vars.bias_0 = jax_initial_vars.logits_ffn.bias.b
  del tf_initial_vars.logits_ffn
  return tf_initial_vars


def replace_jax_conv_bnact_vars_to_tf(jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces JAX ConvBNAct variables to TF compatible variables.

  Args:
    jax_initial_vars: JAX ConvBNAct layer vars.

  Returns:
    tf_initial_vars which is TF compatible with ConvBNAct.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.conv = NestedMap(w=jax_initial_vars.w)
  tf_initial_vars.act = NestedMap()
  del tf_initial_vars.w
  return tf_initial_vars


def replace_jax_res_net_block_vars_to_tf(
    jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces the JAX ResNetBlock vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX ResNetBlock vars.

  Returns:
    tf_initial_vars which is TF compatible with ResNetBlock.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.body = [
      replace_jax_conv_bnact_vars_to_tf(var) for var in tf_initial_vars.body
  ]
  if 'shortcut' in tf_initial_vars:
    tf_initial_vars.shortcut = replace_jax_conv_bnact_vars_to_tf(
        tf_initial_vars.shortcut)
  return tf_initial_vars


def replace_jax_res_net_vars_to_tf(jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces the JAX ResNet vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX ResNet layer vars.

  Returns:
    tf_initial_vars which is TF compatible with ResNet.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.entryflow_conv = replace_jax_conv_bnact_vars_to_tf(
      tf_initial_vars.entryflow_conv)
  stage_id = 0
  block_id = 0
  block = f'stage_{stage_id}_block_{block_id}'
  while block in tf_initial_vars:
    while block in tf_initial_vars:
      tf_initial_vars[block] = replace_jax_res_net_block_vars_to_tf(
          tf_initial_vars[block])
      block_id += 1
      block = f'stage_{stage_id}_block_{block_id}'
    stage_id += 1
    block_id = 0
    block = f'stage_{stage_id}_block_{block_id}'
  return tf_initial_vars


def replace_jax_light_conv_vars_to_tf(jax_initial_vars: NestedMap) -> NestedMap:
  """Replace the JAX LightConv vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX LightConv vars.

  Returns:
    tf_initial_vars which is TF compatible with LightConv.
  """
  tf_initial_vars = py_utils.NestedMap()

  tf_initial_vars.ln = py_utils.NestedMap()
  tf_initial_vars.ln.bias = jax_initial_vars.ln.bias
  tf_initial_vars.ln.scale = jax_initial_vars.ln.scale

  tf_initial_vars.norm = py_utils.NestedMap()
  tf_initial_vars.norm.beta = jax_initial_vars.conv_norm.beta
  tf_initial_vars.norm.gamma = jax_initial_vars.conv_norm.gamma
  tf_initial_vars.norm.moving_mean = jax_initial_vars.conv_norm.moving_mean
  tf_initial_vars.norm.moving_variance = jax_initial_vars.conv_norm.moving_variance

  tf_initial_vars.dropout = [py_utils.NestedMap(), py_utils.NestedMap()]

  tf_initial_vars.depthwise_conv1d = py_utils.NestedMap()
  tf_initial_vars.depthwise_conv1d.w = np.expand_dims(
      jax_initial_vars.depthwise_conv1d.w, axis=-1)

  tf_initial_vars.linear_end = py_utils.NestedMap()
  tf_initial_vars.linear_end.w = jax_initial_vars.linear_end.linear.w
  tf_initial_vars.linear_end.b = jax_initial_vars.linear_end.bias.b

  tf_initial_vars.linear_start = py_utils.NestedMap()
  tf_initial_vars.linear_start.w = np.concatenate([
      jax_initial_vars.linear_start_gated.linear.w,
      jax_initial_vars.linear_start_act.linear.w
  ],
                                                  axis=-1)
  tf_initial_vars.linear_start.b = np.concatenate([
      jax_initial_vars.linear_start_gated.bias.b,
      jax_initial_vars.linear_start_act.bias.b
  ],
                                                  axis=-1)

  tf_initial_vars = to_tf_nmap(tf_initial_vars)
  return tf_initial_vars


def replace_jax_conformer_layer_vars_to_tf(
    jax_initial_vars: NestedMap) -> NestedMap:
  """Replace the JAX conformer layer vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX conformer layer vars.

  Returns:
    tf_initial_vars which is TF compatible with ConformerLayer.
  """

  tf_initial_vars = py_utils.NestedMap()

  tf_initial_vars.lconv = replace_jax_light_conv_vars_to_tf(
      jax_initial_vars.lconv)

  tf_initial_vars.final_ln = py_utils.NestedMap()
  tf_initial_vars.final_ln.bias = jax_initial_vars.final_ln.bias
  tf_initial_vars.final_ln.scale = jax_initial_vars.final_ln.scale

  tf_initial_vars.fflayer_start = py_utils.NestedMap()
  tf_initial_vars.fflayer_start.residual_dropout = jax_initial_vars.fflayer_start.residual_dropout
  tf_initial_vars.fflayer_start.layer_norm = jax_initial_vars.fflayer_start.layer_norm
  tf_initial_vars.fflayer_start.fflayer = py_utils.NestedMap()
  tf_initial_vars.fflayer_start.fflayer.dropout = [
      jax_initial_vars.fflayer_start.relu_dropout, {}
  ]
  tf_initial_vars.fflayer_start.fflayer.fc = [
      py_utils.NestedMap(), py_utils.NestedMap()
  ]
  tf_initial_vars.fflayer_start.fflayer.fc[
      0].w = jax_initial_vars.fflayer_start.ffn_layer1.linear.w
  tf_initial_vars.fflayer_start.fflayer.fc[
      0].b = jax_initial_vars.fflayer_start.ffn_layer1.bias.b
  tf_initial_vars.fflayer_start.fflayer.fc[
      1].w = jax_initial_vars.fflayer_start.ffn_layer2.linear.w
  tf_initial_vars.fflayer_start.fflayer.fc[
      1].b = jax_initial_vars.fflayer_start.ffn_layer2.bias.b

  tf_initial_vars.fflayer_end = py_utils.NestedMap()
  tf_initial_vars.fflayer_end.layer_norm = jax_initial_vars.fflayer_end.layer_norm
  tf_initial_vars.fflayer_end.residual_dropout = jax_initial_vars.fflayer_end.residual_dropout
  tf_initial_vars.fflayer_end.fflayer = py_utils.NestedMap()
  tf_initial_vars.fflayer_end.fflayer.dropout = [
      jax_initial_vars.fflayer_end.relu_dropout, {}
  ]
  tf_initial_vars.fflayer_end.fflayer.fc = [
      py_utils.NestedMap(), py_utils.NestedMap()
  ]
  tf_initial_vars.fflayer_end.fflayer.fc[
      0].w = jax_initial_vars.fflayer_end.ffn_layer1.linear.w
  tf_initial_vars.fflayer_end.fflayer.fc[
      0].b = jax_initial_vars.fflayer_end.ffn_layer1.bias.b
  tf_initial_vars.fflayer_end.fflayer.fc[
      1].w = jax_initial_vars.fflayer_end.ffn_layer2.linear.w
  tf_initial_vars.fflayer_end.fflayer.fc[
      1].b = jax_initial_vars.fflayer_end.ffn_layer2.bias.b

  tf_initial_vars.trans_atten = py_utils.NestedMap()
  tf_initial_vars.trans_atten.layer_norm = jax_initial_vars.trans_atten.norm
  tf_initial_vars.trans_atten.residual_dropout = jax_initial_vars.trans_atten.residual_dropout
  tf_initial_vars.trans_atten.atten = jax_initial_vars.trans_atten.self_atten
  tf_initial_vars = to_tf_nmap(tf_initial_vars)
  return tf_initial_vars
