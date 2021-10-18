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
"""Utility functions for JAX tests."""

from typing import Any, Optional

from absl import flags
from jax import numpy as jnp
from lingvo.jax import py_utils
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS
JTensor = jnp.ndarray
NestedMap = py_utils.NestedMap


def ToNp(x: JTensor) -> np.ndarray:
  """Converts TF/JAX tensors to numpy."""
  return np.asarray(x, dtype=np.float32)


def UnshardInputNmap(x_nmap: NestedMap) -> NestedMap:
  """Unshards input sequences.

  Args:
    x_nmap: NestedMap with some tensors of shape [num_devices, batch, ...].

  Returns:
    NestedMap with tensors reshaped to [num_devices * batch, seq_len].
  """

  def Unshard(x: JTensor) -> JTensor:
    num_devices = x.shape[0]
    batch_size = x.shape[1]
    new_shape = [num_devices * batch_size] + x.shape[2:]
    return tf.reshape(x, new_shape)

  return tf.nest.map_structure(Unshard, x_nmap)


def ToTfNmap(x_nmap: NestedMap) -> NestedMap:
  """Converts numpy dtypes to TF dtypes in a variable collection."""

  def ToTf(x: Any) -> JTensor:
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
      assert 'dtype not supported yet'

  return tf.nest.map_structure(ToTf, x_nmap)


def ReplaceJaxAttentionVarsToTf(
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


def ReplaceJaxSingleShardFullSoftmaxVarsToTf(
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


def ReplaceJaxSimpleFullSoftmaxVarsToTf(
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


def ReplaceJaxConvBNActVarsToTf(jax_initial_vars: NestedMap) -> NestedMap:
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


def ReplaceJaxResNetBlockVarsToTf(jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces the JAX ResNetBlock vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX ResNetBlock vars.

  Returns:
    tf_initial_vars which is TF compatible with ResNetBlock.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.body = [
      ReplaceJaxConvBNActVarsToTf(var) for var in tf_initial_vars.body
  ]
  if 'shortcut' in tf_initial_vars:
    tf_initial_vars.shortcut = ReplaceJaxConvBNActVarsToTf(
        tf_initial_vars.shortcut)
  return tf_initial_vars


def ReplaceJaxResNetVarsToTf(jax_initial_vars: NestedMap) -> NestedMap:
  """Replaces the JAX ResNet vars to TF compatible vars.

  Args:
    jax_initial_vars: JAX ResNet layer vars.

  Returns:
    tf_initial_vars which is TF compatible with ResNet.
  """
  tf_initial_vars = jax_initial_vars.copy()
  tf_initial_vars.entryflow_conv = ReplaceJaxConvBNActVarsToTf(
      tf_initial_vars.entryflow_conv)
  stage_id = 0
  block_id = 0
  block = f'stage_{stage_id}_block_{block_id}'
  while block in tf_initial_vars:
    while block in tf_initial_vars:
      tf_initial_vars[block] = ReplaceJaxResNetBlockVarsToTf(
          tf_initial_vars[block])
      block_id += 1
      block = f'stage_{stage_id}_block_{block_id}'
    stage_id += 1
    block_id = 0
    block = f'stage_{stage_id}_block_{block_id}'
  return tf_initial_vars
