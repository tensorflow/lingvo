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
"""Embedding and softmax layers."""

import math
from typing import Optional, Union

import jax
from jax import numpy as jnp
from lingvo.jax import base_layer
from lingvo.jax import py_utils
from lingvo.jax import pytypes
from lingvo.jax.layers import linears
import numpy as np

NestedMap = py_utils.NestedMap
weight_params = py_utils.weight_params

InstantiableParams = py_utils.InstantiableParams
JTensor = pytypes.JTensor


class SingleShardEmbedding(base_layer.BaseLayer):
  """Embedding layer that is not sharded."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('vocab_size', 0, 'Num tokens in vocab.')
    p.Define('embedding_dims', 0, 'Depth of the output.')
    p.Define('lookup_style', 'index',
             'Style of lookup, one of index or matmul.')
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        ' with sqrt(embedding_dim) in emb_lookup.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    assert p.vocab_size > 0
    assert p.embedding_dims > 0

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    self.create_variable(
        'emb_var',
        weight_params(
            shape=[p.vocab_size, p.embedding_dims],
            init=p.params_init,
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp.wt))

  def fprop(self, ids: JTensor) -> JTensor:
    p = self.params
    theta = self.local_theta()
    if p.lookup_style == 'index':
      embs = jnp.asarray(theta.emb_var)[(ids,)]
    elif p.lookup_style == 'matmul':
      one_hot_ids = jax.nn.one_hot(ids, p.vocab_size)
      embs = jnp.matmul(one_hot_ids, theta.emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    if p.scale_sqrt_depth:
      embs *= p.embedding_dims**0.5
    return embs


class SingleShardFullSoftmax(base_layer.BaseLayer):
  """Softmax layer that is not sharded."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    """Params for SoftmaxLayer."""
    p = super().Params()
    p.Define('input_dims', 0, 'Dimension of the input.')
    p.Define('num_classes', 0, 'Total number of target classes.')
    p.Define('soft_cap_logits', 0.,
             'If not None logits are soft capped to this value.')
    p.Define('bi_tempered_loss', None, 'If not None applies bi-tempered loss.')
    p.Define('label_smoothing_prob', 0.0, 'Label smoothing probability.')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    ff_p = linears.FeedForward.Params().Set(
        input_dims=p.input_dims,
        output_dims=p.num_classes,
        activation='NONE',
        weight_split_dims_mapping=wp.Copy(),
        activation_split_dims_mapping=ap.Copy())
    self.create_child('logits_ffn', ff_p)
    if p.bi_tempered_loss:
      self.create_child('bi_tempered_loss', p.bi_tempered_loss)

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to soft cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    p = self.params
    # Compute logits
    logits = self.logits_ffn.fprop(inputs)

    # Soft cap logits if applicable
    if p.soft_cap_logits:
      logits = p.soft_cap_logits * jnp.tanh(logits / p.soft_cap_logits)
    return logits

  def fprop(self,
            inputs: JTensor,
            class_weights: JTensor,
            class_ids: Optional[JTensor] = None,
            class_probabilities: Optional[JTensor] = None) -> NestedMap:
    """Computes logits, cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] containing the weights for
        each target word.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: with shape [...]. class_weights casted to
        this layer's dtype.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    p = self.params
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), p.num_classes)
      if p.label_smoothing_prob > 0.0:
        # Label smoothing reduce the probability of the label from 1 to
        # 1 - label_smoothing_prob, and redistribute label_smoothing_prob to the
        # rest of num_classes - 1 classes where each class has a probability of
        # label_smoothing_prob / (num_classes - 1).
        other_prob = p.label_smoothing_prob / (p.num_classes - 1)
        class_probabilities = (
            (1.0 - p.label_smoothing_prob) * class_probabilities + other_prob *
            (1.0 - class_probabilities)).astype(self.fprop_dtype)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)

    if p.bi_tempered_loss is None:
      per_example_xent = -jnp.sum(log_probs * class_probabilities, axis=-1)
    else:
      per_example_xent = self.bi_tempered_loss.fprop(logits,
                                                     class_probabilities)
    per_example_argmax = jax.lax.stop_gradient(jnp.argmax(logits, axis=-1))

    # Compute total softmax for the entire sequence
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights)
    total_weight = jnp.sum(class_weights)
    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax.astype(inputs_dtype),
        per_example_xent=per_example_xent.astype(inputs_dtype),
        total_xent=total_xent.astype(inputs_dtype),
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(inputs_dtype))

    return output_nmap


class SingleShardSharedEmbeddingSoftmax(SingleShardFullSoftmax):
  """Softmax layer that is not sharded and supports embedding lookup."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('lookup_style', 'index',
             'Style of lookup, one of index or matmul.')
    p.Define(
        'scale_sqrt_depth', False, 'If set True, activations are scaled'
        'with sqrt(embedding_dim) in emb_lookup.')
    ap = p.activation_split_dims_mapping
    ap.Define('emb_out_split_dims_mapping', None, 'Sharding of the emb output.')
    return p

  def emb_lookup(self, ids: JTensor) -> JTensor:
    p = self.params
    ap = p.activation_split_dims_mapping
    emb_var = jnp.transpose(self.logits_ffn.linear.local_theta().w)
    if p.lookup_style == 'index':
      embs = jnp.asarray(emb_var)[(ids,)]
    elif p.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(ids, p.num_classes, dtype=self.fprop_dtype)
      embs = linears.project_last_dim(one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')
    # Scale with sqrt(embedding dims)
    if p.scale_sqrt_depth:
      embs *= p.input_dims**0.5

    embs = base_layer.maybe_shard(embs, ap.emb_out_split_dims_mapping,
                                  p.mesh_axis_names)
    return embs


class GShardSharedEmbeddingSoftmax(base_layer.BaseLayer):
  """Softmax layer with embedding lookup and Gaussian init used in gshard.

  Features:
  1) Weight shape is [V, M] where V is num_classes and M is input_dims.
  2) No bias
  3) Apply 1/sqrt(M) to the input activations before computing the logits.
  4) Optionally using soft clipping and absolute value clipping of logits.
  """

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('input_dims', 0, 'Dimension of the input.')
    p.Define('num_classes', 0, 'Total number of target classes.')
    p.Define(
        'use_tgt_labels_size_as_loss_denominator', True,
        'False to use total number of non-padding tokens instead of '
        'fixed tgt_labels tensor size.')
    # logits_abs_max = 20 for m4_hybrid model
    p.Define(
        'soft_cap_logits', 0.,
        'If not None logits are soft capped to this value before '
        ' the absolute value clipping with p.logits_abs_max.')
    p.Define('logits_abs_max', None, 'Absolute logits clipping.')
    p.Define(
        'z_loss_weight', 0, 'if z_loss_weight is nonzero, we add a loss equal '
        'to z_loss_weight * square(logsumexp(logits, -1))')
    ap = p.activation_split_dims_mapping
    ap.Define('emb_out_split_dims_mapping', None,
              'Mesh split for embedding outputs..')
    return p

  def __init__(self, params: InstantiableParams) -> None:
    super().__init__(params)
    p = self.params
    wp = p.weight_split_dims_mapping
    ap = p.activation_split_dims_mapping
    emb_p = linears.Linear.Params().Set(
        input_dims=p.num_classes,
        output_dims=p.input_dims,
        # Same as in gshard_builder.DenseBuilder.Embedding
        params_init=py_utils.WeightInit.Gaussian(),
        weight_split_dims_mapping=wp.Copy(),
        activation_split_dims_mapping=ap.Copy())
    self.create_child('embedding', emb_p)

  def emb_lookup(self, ids: JTensor) -> JTensor:
    p = self.params
    ap = p.activation_split_dims_mapping
    # BL -> BLV
    one_hot_ids = jax.nn.one_hot(ids, p.num_classes, dtype=self.fprop_dtype)
    # BLV,VH -> BLH
    embs = linears.project_last_dim(one_hot_ids, self.embedding.local_theta().w)
    embs = base_layer.maybe_shard(embs, ap.emb_out_split_dims_mapping,
                                  p.mesh_axis_names)
    return embs

  def get_logits(self, inputs: JTensor) -> JTensor:
    """Returns logits given the inputs with an option to cap it.

    Args:
      inputs: a single JTensor with shape [..., input_dim].

    Returns:
      logits: with shape [..., num_classes]. Unnormalized softmax's logits.
    """
    p = self.params
    ap = p.activation_split_dims_mapping
    # activations are scaled with 1/sqrt(input_dims)
    inputs *= (p.input_dims**-0.5)
    # VH -> HV
    softmax_var = jnp.transpose(self.embedding.local_theta().w)
    # Compute logits:  BLH,HV -> BLV
    logits = linears.project_last_dim(inputs, softmax_var)
    logits = base_layer.maybe_shard(logits, ap.out, p.mesh_axis_names)

    # Soft cap logits if applicable
    if p.soft_cap_logits:
      logits = p.soft_cap_logits * jnp.tanh(logits / p.soft_cap_logits)

    # abs cap logits if applicable
    if p.logits_abs_max:
      logits = jnp.clip(logits, -p.logits_abs_max, p.logits_abs_max)
    return logits

  def compute_z_loss(self, logits):
    """Returns a z_loss regularization which stablize logits."""
    # Applies stop_gradient to max_logit instead of logits.
    max_logit = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    exp_x = jnp.exp(logits - max_logit)
    sum_exp_x = jnp.sum(exp_x, axis=-1, keepdims=True)
    log_z = jnp.log(sum_exp_x) + max_logit
    return jnp.square(log_z)

  def fprop(self,
            inputs: JTensor,
            class_weights: JTensor,
            class_ids: Optional[JTensor] = None,
            class_probabilities: Optional[JTensor] = None) -> NestedMap:
    """Computes logits, cross entropy etc.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      class_weights: a JTensor with shape [..., 1] containing the weights for
        each target word.
      class_ids: a JTensor with shape [..., 1] of int32 dtype containing the
        target class labels.
      class_probabilities: a JTensor with shape [..., num_classes] of float
        values indicating class-membership probabilities.

    Returns:
      A `.NestedMap` containing the following fields

      - logits: with shape [..., num_classes]. Unnormalized softmax's logits.
      - per_example_argmax: with shape [...]. argmax of i-th example.
      - per_example_xent: with shape [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: with shape [...]. class_weights casted to
        this layer's dtype.
      - total_xent: A scalar. The sum of per_example_weight * per_example_xent.
      - total_weight: A scalar. The sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
    """
    p = self.params
    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    # Compute logits
    inputs_dtype = inputs.dtype
    logits = self.get_logits(inputs)
    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)

    if class_probabilities is None:
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1), p.num_classes)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)

    per_example_xent = -jnp.sum(log_probs * class_probabilities, axis=-1)

    per_example_argmax = jax.lax.stop_gradient(jnp.argmax(logits, axis=-1))

    # Compute total softmax for the entire sequence
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights)

    total_weight = jnp.sum(class_weights)

    if p.use_tgt_labels_size_as_loss_denominator:
      loss_denominator = jnp.sum(jnp.ones_like(class_weights))
    else:
      loss_denominator = total_weight
    avg_xent = (total_xent / loss_denominator).astype(inputs_dtype)
    z_loss = (
        jnp.sum(self.compute_z_loss(logits) * class_weights) / loss_denominator)
    z_loss *= p.z_loss_weight
    base_layer.add_summary('aux_z_loss', z_loss)
    aux_loss_ctx = py_utils.AuxLossContext.Current()

    if aux_loss_ctx is not None:
      aux_loss_ctx.AddLoss(z_loss)

    output_nmap = NestedMap(
        logits=logits.astype(inputs_dtype),
        log_probs=log_probs.astype(inputs_dtype),
        per_example_argmax=per_example_argmax.astype(inputs_dtype),
        per_example_xent=per_example_xent.astype(inputs_dtype),
        total_xent=total_xent.astype(inputs_dtype),
        # base_model.py _compute_xent_loss_helper uses avg_xent_weight if set,
        # this helper is currently used by LanguageModel only, if we have
        # EncoderDecoder model we will have to adjust weighting as well.
        avg_xent_weight=loss_denominator,
        avg_xent=avg_xent,
        total_weight=total_weight)

    return output_nmap


class PositionalEmbedding(base_layer.BaseLayer):
  """Generates position embedding for a given 1-d sequence."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define(
        'min_timescale', 1, 'Start of the geometric index.'
        'Determines the periodicity of the added signal.')
    p.Define(
        'max_timescale', 1e4, 'End of the geometric index. '
        'Determines the frequency of the added signal.')
    p.Define('embedding_dims', 0, 'Dimension of the embedding to be generated.')
    return p

  def fprop(self,
            seq_length: Optional[int] = None,
            position: Optional[JTensor] = None) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      seq_length: Sequence length of the embeddings to be generated. This may be
        omitted if an explicit position JTensor is specified.
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [batch, seq_length].

    Returns:
      a JTensor of shape [batch, seq_length, embedding_dim] if position JTensor
      is specified, else of shape [1, seq_length, embedding_dim].
    """
    p = self.params
    if position is None:
      assert seq_length is not None
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    num_timescales = p.embedding_dims // 2
    log_timescale_increment = (
        math.log(float(p.max_timescale) / float(p.min_timescale)) /
        jnp.maximum(jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1))
    inv_timescales = p.min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) *
        -log_timescale_increment)
    scaled_time = (
        position[:, :, jnp.newaxis] *
        inv_timescales[jnp.newaxis, jnp.newaxis, :])
    signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)],
        axis=2).astype(self.fprop_dtype)
    # Force usage of `np` rather than `jnp` to compute static values at trace
    # time.
    signal = jnp.pad(signal, [[0, 0], [0, 0], [0, np.mod(p.embedding_dims, 2)]])
    return signal


class RotaryPositionalEmbedding(PositionalEmbedding):
  """Applies rotary position embedding for a given 1-d sequence.

  The Rotary position embedding is described in https://arxiv.org/abs/2104.09864
  """

  def fprop(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      inputs: JTensor,
      position: Optional[JTensor] = None,
  ) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a JTensor of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    p = self.params
    if len(inputs.shape) != 4:
      raise ValueError('Input is assumed to be a rank 4 tensor of shape'
                       '[batch, sequence, heads, dims].')
    if p.embedding_dims % 2:
      raise ValueError('Embedding dim for rotary position embedding must be a'
                       'multiple of 2.')
    if p.embedding_dims != inputs.shape[3]:
      raise ValueError('The embedding dims of the rotary position embedding'
                       'must match the hidden dimension of the inputs.')
    half_embedding_dim = p.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / p.embedding_dims
    timescale = p.min_timescale * (p.max_timescale / p.min_timescale)**fraction
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return jnp.concatenate([first_part, second_part], axis=-1)

  def extend_step(self,
                  inputs: JTensor,
                  time_step: Optional[Union[int, JTensor]] = None) -> JTensor:
    """Generates a JTensor of sinusoids with different frequencies for a step.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, N, H] or of shape [B,
        P, N, H] where P may be a prefix length.
      time_step: The time step which is being decoded, this should correspond to
        the time step of the last token in the prefix window (P) in the entire
        sequence length S.

    Returns:
      a JTensor of the same shape as input with the rotary position embedding
      incorporated in it.
    """
    assert len(inputs.shape) in [3, 4]
    inputs_shape = inputs.shape
    if len(inputs_shape) == 3:
      inputs = inputs[:, jnp.newaxis, :, :]
    seq_length = inputs.shape[1]
    # Adjust the position with the time step.
    # Note that time_step may be a tracer rather than an int, and so we must
    # use jax.lax.iota, rather than jnp.arange.
    position = jax.lax.iota(dtype=jnp.int32, size=seq_length)
    position = time_step - jnp.flip(position)
    position = jnp.where(position < 0, jnp.zeros_like(position), position)
    output = self.fprop(inputs, position=position[jnp.newaxis, :])
    if len(inputs_shape) == 3:
      output = jnp.squeeze(output, axis=1)
    return output


class TrainablePositionalEmbedding(PositionalEmbedding):
  """Generates trainable position embedding for a given 1-d sequence."""

  @classmethod
  def Params(cls) -> InstantiableParams:
    p = super().Params()
    p.Define('max_seq_length', 10240, 'Max sequence length.')
    p.Define('lookup_style', 'matmul',
             'Style of lookup, one of index or matmul.')
    return p

  def create_layer_variables(self) -> None:
    super().create_layer_variables()
    p = self.params
    wp = p.weight_split_dims_mapping
    self.create_variable(
        'emb_var',
        weight_params(
            shape=[p.max_seq_length, p.embedding_dims],
            init=p.params_init,
            dtype=p.dtype,
            device_mesh=p.device_mesh,
            tensor_split_dims_mapping=wp.wt))

  def fprop(self,
            seq_length: Optional[int] = None,
            position: Optional[JTensor] = None) -> JTensor:
    """Generates a JTensor of embedding lookup result.

    Args:
      seq_length: Sequence length of the embeddings to be generated. This may be
        omitted if an explicit position JTensor is specified.
      position: Optional position JTensor which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [batch, seq_length].

    Returns:
      a JTensor of shape [batch, seq_length, embedding_dim] if position JTensor
      is specified, else of shape [1, seq_length, embedding_dim].
    """
    p = self.params
    theta = self.local_theta()
    if position is None:
      assert seq_length is not None
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    pos_emb_var = theta.emb_var
    pos_emb_var = jax.lax.slice_in_dim(pos_emb_var, 0, seq_length, axis=0)
    if p.lookup_style == 'index':
      embs = jnp.asarray(pos_emb_var)[(position,)]
    elif p.lookup_style == 'matmul':
      one_hot_ids = jax.nn.one_hot(position, seq_length)
      embs = jnp.matmul(one_hot_ids, pos_emb_var)
    else:
      raise ValueError('Unknown lookup style.')

    return embs
