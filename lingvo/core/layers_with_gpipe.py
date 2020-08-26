# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Lingvo layers that depend on layers and gpipe."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import batch_major_attention
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import tshape
from lingvo.core.gpipe import FeatureExtractionLayer
from lingvo.core.gpipe import PipeliningLayer


def _common_gpipe_transformer_params(p):
  """Add GPipe params to layer."""
  p.Define(
      'is_transparent', False,
      'If set, encoder outputs a list of layer outputs while decoder '
      'expects a list of source input vectors.')
  p.Define('transparent_merger_tpl', None,
           'Creates weights for transparent combination.')
  p.Define(
      'final_enc_layer', False,
      'True for final encoder layer. To be used for final transparent merger.')
  p.Define(
      'normalize_output', False,
      'If set, encoder outputs a list of layer outputs while decoder '
      'expects a list of source input vectors.')
  p.Define('ln_tpl', layers.LayerNorm.Params(), 'Layer norm default params')
  return p


def _common_gpipe_transformer_init(layer):
  """Initialize a GPipe layer."""
  p = layer.params

  if p.normalize_output:
    params = p.ln_tpl.Copy()
    params.name = 'encoder_ln'
    params.input_dim = p.source_dim
    layer.CreateChild('layer_norm', params)

  if p.is_transparent and p.transparent_merger_tpl is not None:
    transparent_param = p.transparent_merger_tpl.Copy()
    transparent_param.name = 'transparent_0'
    layer.CreateChild('transparent_merger', transparent_param)
  assert p.name


def _common_gpipe_transformer_encoder_fprop(
    layer, layer_class, theta, source_vecs, source_paddings, target_vecs,
    target_paddings, source_segment_id, target_segment_id, transparent_acc,
    transparent_acc_helper, source_task_id, target_task_id):
  """GPipe encoder FProp."""
  p = layer.params
  if source_task_id is not None or target_task_id is not None:
    h, _ = super(layer_class, layer).FProp(
        theta,
        source_vecs,
        source_paddings,
        source_segment_id=source_segment_id,
        source_task_id=source_task_id,
        target_task_id=target_task_id)
  else:
    h, _ = super(layer_class, layer).FProp(
        theta,
        source_vecs,
        source_paddings,
        source_segment_id=source_segment_id)
  h.set_shape(source_vecs.shape)
  if p.is_transparent:
    if p.transparent_merger_tpl is not None:
      transparent_acc_helper = layer.transparent_merger.FProp(
          theta.transparent_merger)
      transparent_acc = tf.zeros_like(source_vecs)
    transparent_acc = transparent_acc + transparent_acc_helper[0] * source_vecs
    if p.final_enc_layer:
      h = transparent_acc + h * transparent_acc_helper[-1]
      transparent_acc = None
      transparent_acc_helper = None
    else:
      transparent_acc_helper = transparent_acc_helper[1:]
  if p.normalize_output:
    h = layer.layer_norm.FProp(theta.layer_norm, h)

  if source_task_id is not None or target_task_id is not None:
    return (h, source_paddings, target_vecs, target_paddings, source_segment_id,
            target_segment_id, transparent_acc, transparent_acc_helper,
            source_task_id, target_task_id)
  else:
    return (h, source_paddings, target_vecs, target_paddings, source_segment_id,
            target_segment_id, transparent_acc, transparent_acc_helper)


def _common_gpipe_transformer_decoder_fprop(
    layer, layer_class, theta, source_vecs, source_paddings, target_vecs,
    target_paddings, source_segment_id, target_segment_id, transparent_acc,
    transparent_acc_helper, source_task_id, target_task_id):
  """GPipe decoder FProp."""
  assert target_vecs is not None
  assert target_paddings is not None
  if source_task_id is not None or target_task_id is not None:
    h, _ = super(layer_class, layer).FProp(
        theta,
        target_vecs,
        target_paddings,
        aux_vecs=source_vecs,
        aux_paddings=source_paddings,
        source_segment_id=target_segment_id,
        aux_segment_id=source_segment_id,
        source_task_id=source_task_id,
        target_task_id=target_task_id)
  else:
    h, _ = super(layer_class, layer).FProp(
        theta,
        target_vecs,
        target_paddings,
        aux_vecs=source_vecs,
        aux_paddings=source_paddings,
        source_segment_id=target_segment_id,
        aux_segment_id=source_segment_id)
  h.set_shape(target_vecs.shape)

  if source_task_id is not None or target_task_id is not None:
    return (source_vecs, source_paddings, h, target_paddings, source_segment_id,
            target_segment_id, transparent_acc, transparent_acc_helper,
            source_task_id, target_task_id)
  else:
    return (source_vecs, source_paddings, h, target_paddings, source_segment_id,
            target_segment_id, transparent_acc, transparent_acc_helper)


def _common_gpipe_transformer_fprop_meta(p, inputs, *args):
  """GPipe FPropMeta function."""
  # TODO(huangyp): return accurate estimate of flops.
  py_utils.CheckShapes((inputs,))
  flops_per_element = 5
  src_time, source_batch, dim = inputs
  flops = flops_per_element * src_time * src_time * source_batch * dim
  args = args if isinstance(args, tuple) else (args,)
  if not p.has_aux_atten and p.is_transparent:  # Transparent Encoder FPropMeta
    if p.transparent_merger_tpl is not None:
      args = args[:5] + (inputs,
                         tshape.Shape([p.transparent_merger_tpl.num_sources]))
    args = args[:6] + (tshape.Shape([args[6][0] - 1]),)
    if p.final_enc_layer:
      args = args[:5] + (None, None)
  return py_utils.NestedMap(flops=flops, out_shapes=(inputs,) + args)


class GPipeTransformerLayer(layers_with_attention.TransformerLayer):
  """GPipe compatible transformer layer."""

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super().Params()
    return _common_gpipe_transformer_params(p)

  def __init__(self, params):
    super().__init__(params)
    _common_gpipe_transformer_init(self)

  @classmethod
  def SetupDeterministicDropout(cls, params):
    """Replaced dropout layers in transformer with deterministic ones."""
    params.tr_atten_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    params.tr_atten_tpl.atten_tpl.atten_dropout_deterministic = True
    params.tr_atten_tpl.atten_tpl.inner_atten_params \
    .atten_dropout_deterministic = True
    params.tr_fflayer_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    params.tr_fflayer_tpl.fflayer_tpl.dropout = (
        layers.DeterministicDropoutLayer.Params())
    return params

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs,
            target_paddings,
            source_segment_id,
            target_segment_id,
            transparent_acc,
            transparent_acc_helper,
            source_task_id=None,
            target_task_id=None):
    p = self.params
    with tf.name_scope(p.name):
      if p.has_aux_atten:  # Decoder FProp
        return _common_gpipe_transformer_decoder_fprop(
            self, GPipeTransformerLayer, theta, source_vecs, source_paddings,
            target_vecs, target_paddings, source_segment_id, target_segment_id,
            transparent_acc, transparent_acc_helper, source_task_id,
            target_task_id)
      else:  # Encoder FProp
        return _common_gpipe_transformer_encoder_fprop(
            self, GPipeTransformerLayer, theta, source_vecs, source_paddings,
            target_vecs, target_paddings, source_segment_id, target_segment_id,
            transparent_acc, transparent_acc_helper, source_task_id,
            target_task_id)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    return _common_gpipe_transformer_fprop_meta(p, inputs, *args)


class GPipeEvolvedTransformerEncoderLayer(
    layers_with_attention.EvolvedTransformerEncoderLayer):
  """GPipe-compatible Evolved Transformer encoder layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    return _common_gpipe_transformer_params(p)

  def __init__(self, params):
    super().__init__(params)
    _common_gpipe_transformer_init(self)

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs,
            target_paddings,
            source_segment_id,
            target_segment_id,
            transparent_acc,
            transparent_acc_helper,
            source_task_id=None,
            target_task_id=None):
    with tf.name_scope(self.params.name):
      return _common_gpipe_transformer_encoder_fprop(
          self, GPipeEvolvedTransformerEncoderLayer, theta, source_vecs,
          source_paddings, target_vecs, target_paddings, source_segment_id,
          target_segment_id, None, None, source_task_id, target_task_id)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    return _common_gpipe_transformer_fprop_meta(p, inputs, *args)

  @classmethod
  def _AttentionSetupDeterministicDropout(cls, tr_atten_tpl):
    tr_atten_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    tr_atten_tpl.atten_tpl.atten_dropout_deterministic = True
    tr_atten_tpl.atten_tpl.inner_atten_params.atten_dropout_deterministic = True

  @classmethod
  def _TransformerSetupDeterministicDropout(cls, transformer_tpl):
    cls._AttentionSetupDeterministicDropout(transformer_tpl.tr_atten_tpl)
    transformer_tpl.tr_fflayer_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    transformer_tpl.tr_fflayer_tpl.fflayer_tpl.dropout = (
        layers.DeterministicDropoutLayer.Params())

  @classmethod
  def SetupDeterministicDropout(cls, params):
    """Replaces dropout layers in ET with deterministic ones."""
    cls._TransformerSetupDeterministicDropout(params.transformer_tpl)
    params.branched_convs_tpl.dropout_tpl = \
        layers.DeterministicDropoutLayer.Params()
    if hasattr(params, 'glu_tpl'):
      params.glu_tpl.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    if hasattr(params, 'tr_atten_tpl'):
      cls._AttentionSetupDeterministicDropout(params.tr_atten_tpl)
    if hasattr(params, 'tr_double_heads_atten_tpl'):
      cls._AttentionSetupDeterministicDropout(params.tr_double_heads_atten_tpl)
    return params


class GPipeEvolvedTransformerDecoderLayer(
    layers_with_attention.EvolvedTransformerDecoderLayer):
  """GPipe-compatible Evolved Transformer decoder layer."""

  @classmethod
  def Params(cls):
    p = super().Params()
    return _common_gpipe_transformer_params(p)

  def __init__(self, params):
    super().__init__(params)
    _common_gpipe_transformer_init(self)

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs,
            target_paddings,
            source_segment_id,
            target_segment_id,
            transparent_acc,
            transparent_acc_helper,
            source_task_id=None,
            target_task_id=None):
    with tf.name_scope(self.params.name):
      return _common_gpipe_transformer_decoder_fprop(
          self, GPipeEvolvedTransformerDecoderLayer, theta, source_vecs,
          source_paddings, target_vecs, target_paddings, source_segment_id,
          target_segment_id, transparent_acc, transparent_acc_helper,
          source_task_id, target_task_id)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    return _common_gpipe_transformer_fprop_meta(p, inputs, *args)

  @classmethod
  def _AttentionSetupDeterministicDropout(cls, tr_atten_tpl):
    tr_atten_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    tr_atten_tpl.atten_tpl.atten_dropout_deterministic = True
    tr_atten_tpl.atten_tpl.inner_atten_params.atten_dropout_deterministic = True

  @classmethod
  def _TransformerSetupDeterministicDropout(cls, transformer_tpl):
    cls._AttentionSetupDeterministicDropout(transformer_tpl.tr_atten_tpl)
    transformer_tpl.tr_fflayer_tpl.residual_dropout_tpl = (
        layers.DeterministicDropoutLayer.Params())
    transformer_tpl.tr_fflayer_tpl.fflayer_tpl.dropout = (
        layers.DeterministicDropoutLayer.Params())

  @classmethod
  def SetupDeterministicDropout(cls, params):
    """Replaces dropout layers in ET with deterministic ones."""
    cls._TransformerSetupDeterministicDropout(params.transformer_tpl)
    params.branched_convs_tpl.dropout_tpl = \
        layers.DeterministicDropoutLayer.Params()
    if hasattr(params, 'glu_tpl'):
      params.glu_tpl.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    if hasattr(params, 'tr_atten_tpl'):
      cls._AttentionSetupDeterministicDropout(params.tr_atten_tpl)
    if hasattr(params, 'tr_double_heads_atten_tpl'):
      cls._AttentionSetupDeterministicDropout(params.tr_double_heads_atten_tpl)
    return params


class GPipeTransformerSoftmaxLayer(layers.SimpleFullSoftmax):
  """GPipe compatible softmax layer for transformers for computing logits."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('inputs_from_decoder', False,
             'Bool, whether inputs to this layer come from decoder or not.')
    return p

  def FProp(self,
            theta,
            source_vecs,
            source_paddings,
            target_vecs,
            target_paddings,
            source_segment_id,
            target_segment_id,
            transparent_acc,
            transparent_acc_helper,
            source_task_id=None,
            target_task_id=None):
    del source_task_id
    del target_task_id
    p = self.params
    if p.inputs_from_decoder:
      transformer_output = target_vecs
    else:
      transformer_output = source_vecs
    dim1, dim2 = tf.shape(transformer_output)[0], tf.shape(
        transformer_output)[1]
    softmax_input = tf.reshape(transformer_output, [-1, p.input_dim])
    output_shape = [dim1, dim2, p.num_classes]
    return tf.reshape(super().Logits(theta, [softmax_input]), output_shape)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    dim1, dim2 = args[1][:2] if p.inputs_from_decoder else inputs[:2]
    logits = tshape.Shape([dim1, dim2, p.num_classes])
    return py_utils.NestedMap(flops=100, out_shapes=(logits,))


class GPipeTransformerEmbeddingLayer(base_layer.BaseLayer):
  """GPipe compatible embeddings for transformers."""

  @classmethod
  def Params(cls):
    """Configs of Embedding layers for TransformerStack."""
    p = super().Params()
    # Note: we use the same configs for src and tgt embeddings right now.
    p.Define('token_emb', layers.SimpleEmbeddingLayer.Params(),
             'The embedding layer params.')
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'dropout_tpl', layers.DropoutLayer.Params(),
        'Replace with deterministic dropout for splits > 1 '
        'or microbatches > 1.')
    p.Define('add_tgt_embedding_layer', False,
             'Set True if layer embeds tgt instead of src.')
    p.Define('packed_input', False, 'Set True to support packed inputs.')
    p.Define(
        'is_transparent', False,
        'If set, encoder outputs a list of layer outputs while decoder '
        'expects a list of source input vectors.')
    p.Define('max_seq_len', 300, 'Max. seq len for decoding.')
    p.Define('target_vocab_size', 0, 'Target vocab size, if different.')

    # Supporting task embeddings as additional input.
    p.Define('dec_task_emb', None,
             'Adds task embeddings to every decoder timestep.')
    p.Define('enc_task_emb', None,
             'Adds task embeddings to every encoder timestep.')
    p.Define('batch_dim', 1, 'The batch dimension.')
    p.Define('ret_task_ids', False,
             'Includes src_task_id and tgt_id in the fprop returns')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    p.token_emb.name = 'src_token_emb'
    p.position_emb.name = 'src_position_emb'
    self.CreateChild('src_token_emb', p.token_emb)
    self.CreateChild('src_pos_emb', p.position_emb)
    if p.enc_task_emb:
      self.CreateChild('src_task_emb', p.enc_task_emb)

    p.dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    p.dropout_tpl.name = 'src_dropout'
    self.CreateChild('src_dropout', p.dropout_tpl)

    if p.add_tgt_embedding_layer:
      params = p.token_emb.Copy()
      if p.target_vocab_size:
        params.vocab_size = p.target_vocab_size
      params.name = 'tgt_token_emb'
      self.CreateChild('tgt_token_emb', params)
      params = p.position_emb.Copy()
      params.name = 'tgt_position_emb'
      self.CreateChild('tgt_pos_emb', params)
      if p.dec_task_emb:
        self.CreateChild('tgt_task_emb', p.dec_task_emb)

      params = p.dropout_tpl.Copy()
      params.keep_prob = (1.0 - p.input_dropout_prob)
      params.name = 'tgt_dropout'
      self.CreateChild('tgt_dropout', params)

  def GetEmbeddings(self, emb_theta, emb, pos_emb_theta, pos_emb, dropout_theta,
                    dropout, input_ids, input_segment_pos, task_emb_theta,
                    task_emb, task_ids):
    p = self.params
    time_dim = 0 if p.batch_dim else 1
    seq_len = tf.shape(input_ids)[time_dim]
    input_embs = emb.EmbLookup(emb_theta, input_ids)
    if p.packed_input:  # Packed inputs.
      pos_embs = pos_emb.FPropWithPosition(pos_emb_theta, input_segment_pos)
    else:
      pos_embs = tf.expand_dims(
          pos_emb.FProp(pos_emb_theta, seq_len), p.batch_dim)

    input_embs += pos_embs
    if task_emb:
      input_embs += task_emb.EmbLookup(task_emb_theta, task_ids)
    input_embs = dropout.FProp(dropout_theta, input_embs)
    return input_embs

  # To be used for decoding.
  def GetEncoderEmbeddingsDefaultTheta(self, input_ids, task_ids=None):
    p = self.params
    time_dim = 0 if p.batch_dim else 1
    seq_len = tf.shape(input_ids)[time_dim]
    input_embs = self.src_token_emb.EmbLookup(self.theta.src_token_emb,
                                              input_ids)
    pos_embs = tf.expand_dims(
        self.src_pos_emb.FProp(self.theta.src_pos_emb, seq_len), p.batch_dim)
    input_embs += pos_embs
    if task_ids is not None and p.enc_task_emb:
      input_embs += self.src_task_emb.EmbLookup(self.theta.src_task_emb,
                                                task_ids)
    input_embs = self.src_dropout.FProp(self.theta.src_dropout, input_embs)
    return input_embs

  # To be used for decoding.
  def GetDecoderEmbeddingsDefaultTheta(self, input_ids, task_ids=None, t=None):
    p = self.params
    input_embs = self.tgt_token_emb.EmbLookup(self.theta.tgt_token_emb,
                                              input_ids)
    if t is None:
      time_dim = 0 if p.batch_dim else 1
      seq_len = tf.shape(input_ids)[time_dim]
      pos_embs = tf.expand_dims(
          self.tgt_pos_emb.FProp(self.theta.tgt_pos_emb, seq_len), p.batch_dim)
    else:  # Support decoding.
      pos_embs = tf.slice(
          self.tgt_pos_emb.FProp(self.theta.tgt_pos_emb, p.max_seq_len), [t, 0],
          [1, p.token_emb.embedding_dim])
    input_embs += pos_embs
    if task_ids is not None and p.dec_task_emb:
      input_embs += self.tgt_task_emb.EmbLookup(self.theta.tgt_task_emb,
                                                task_ids)
    input_embs = self.tgt_dropout.FProp(self.theta.tgt_dropout, input_embs)
    return input_embs

  def FProp(self, theta, source_id, source_paddings, target_id, target_paddings,
            source_segment_id, target_segment_id, source_segment_pos,
            target_segment_pos, source_task_id, target_task_id):
    p = self.params
    with tf.name_scope(p.name):
      src_task_emb, src_task_emb_theta = None, None
      if p.enc_task_emb:
        src_task_emb, src_task_emb_theta = self.src_task_emb, theta.src_task_emb
      source_vecs = self.GetEmbeddings(theta.src_token_emb, self.src_token_emb,
                                       theta.src_pos_emb, self.src_pos_emb,
                                       theta.src_dropout, self.src_dropout,
                                       source_id, source_segment_pos,
                                       src_task_emb_theta, src_task_emb,
                                       source_task_id)
      target_vecs = None
      if p.add_tgt_embedding_layer:
        tgt_task_emb, tgt_task_emb_theta = None, None
        if p.enc_task_emb:
          tgt_task_emb, tgt_task_emb_theta = (self.tgt_task_emb,
                                              theta.tgt_task_emb)
        target_vecs = self.GetEmbeddings(theta.tgt_token_emb,
                                         self.tgt_token_emb, theta.tgt_pos_emb,
                                         self.tgt_pos_emb, theta.tgt_dropout,
                                         self.tgt_dropout, target_id,
                                         target_segment_pos, tgt_task_emb_theta,
                                         tgt_task_emb, target_task_id)
      rets = (source_vecs, source_paddings, target_vecs, target_paddings,
              source_segment_id, target_segment_id, None, None)
      rets += (source_task_id, target_task_id) if p.ret_task_ids else ()
      return rets

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    # TODO(ankurbpn): return accurate estimate of flops.
    py_utils.CheckShapes((inputs,))
    flops_per_element = 2  # Is this correct?
    vocab = p.token_emb.vocab_size
    dim = p.token_emb.embedding_dim
    src_dim_0, src_dim_1 = inputs
    flops = flops_per_element * src_dim_0 * src_dim_1 * dim * vocab
    args = args if isinstance(args, tuple) else (args,)
    new_inputs = tshape.Shape([src_dim_0, src_dim_1, dim])
    new_args = list(args)
    if p.add_tgt_embedding_layer:
      tgt_dim_0, tgt_dim_1 = args[1]
      new_args[1] = tshape.Shape([tgt_dim_0, tgt_dim_1, dim])
    if p.ret_task_ids:
      new_args = new_args[:5] + [None, None] + new_args[7:]
    else:
      new_args = new_args[:5] + [None, None]
    new_args = tuple(new_args)
    return py_utils.NestedMap(flops=flops, out_shapes=(new_inputs,) + new_args)


# TODO(ankurbpn,huangyp): Deprecate support for batch major layers here.
class GPipeTransformerStack(PipeliningLayer):
  """Stacked self- multi-head attention and fully connected layers.

  With optional layer normalization applied to the final output.

  See 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
  for details.

  The use of this stack for batch major transformer is deprecated.
  Use GPipeBatchMajorTransformerStack instead.
  """

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super().Params()

    # GPipe Related
    p.Define(
        'splits', 1,
        'Number of splits, or list of integers specifying the ending index for '
        'each split in ascending order. Last index should be num_layers.')

    # Transformer related
    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('num_encoder_layers', 0, 'Number of transformer encoder layers.')
    p.Define('num_decoder_layers', 0, 'Number of transformer encoder layers.')
    p.Define('use_pipelined_embeddings', True, 'Deprecated.')
    p.Define('emb_tpl', GPipeTransformerEmbeddingLayer.Params(),
             'Prepare embeddings for Transformer input.')
    p.Define('softmax_tpl', GPipeTransformerSoftmaxLayer.Params(),
             'Optional softmax layer to compute the logits.')
    p.Define('label_smoothing', None, 'Label smoothing Params.')
    p.Define('encoder_tpl', GPipeTransformerLayer.Params(),
             'TransformerLayer Encoder params tpl.')
    p.Define('decoder_tpl', GPipeTransformerLayer.Params(),
             'TransformerLayer Decoder params tpl.')
    p.Define('transparent_merger_dropout_prob', 0.1,
             'Dropout probability in WeightedSumLayer')
    p.Define(
        'is_transparent', False,
        'If set, encoder outputs a merger of embeddings and '
        'layer outputs.')
    p.Define('transparent_merger_tpl', DeterministicWeightsLayer.Params(),
             'Creates weights for transparent combination.')
    p.Define('packed_input', False,
             'If True, assumes multiple training samples per input.')
    p.Define('normalize_encoder', False,
             'If True, layer-normalizes final encoder layer output.')
    p.encoder_tpl.has_aux_atten = False
    p.decoder_tpl.has_aux_atten = True
    p.decoder_tpl.mask_self_atten = True
    p.batch_dim = 1
    return p

  def __init__(self, params):
    p = params.Copy()
    num_layers = p.num_encoder_layers + p.num_decoder_layers

    if isinstance(p.splits, (list, tuple)):
      assert p.splits[-1] == num_layers
      for i, j in zip(p.splits[:-1], p.splits[1:]):
        assert i <= j, 'Splits must be in increasing order.'
    else:
      num_splits = p.splits
      layers_per_split = (num_layers - 1) // num_splits + 1
      p.splits = []
      for i in range(num_splits):
        p.splits.append((i + 1) * layers_per_split)
      p.splits[-1] = num_layers

    transformers = []

    if p.is_transparent:
      p.transparent_merger_tpl.num_sources = p.num_encoder_layers + 1
      p.transparent_merger_tpl.dropout_tpl.keep_prob = (
          1 - p.transparent_merger_dropout_prob)

    # Encoder Embedding layer.
    if len(p.splits) > 1 or p.num_micro_batches > 1:
      p.emb_tpl.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    p.emb_tpl.packed_input = p.packed_input
    p.emb_tpl.is_transparent = p.is_transparent
    p.emb_tpl.add_tgt_embedding_layer = (p.num_decoder_layers > 0)
    p.emb_tpl.name = 'emb'
    p.emb_tpl.batch_dim = p.batch_dim
    transformers.append(p.emb_tpl)
    if p.softmax_tpl:
      p.softmax_tpl.name = 'softmax'
      p.softmax_tpl.inputs_from_decoder = p.num_decoder_layers > 0
    # Encoder layers.
    for i in range(p.num_encoder_layers):
      params = p.encoder_tpl.Copy()
      params.name = 'encoder_%d' % i
      if p.is_transparent:
        params.is_transparent = p.is_transparent
        params.final_enc_layer = (i == (p.num_encoder_layers - 1))
      if p.normalize_encoder and (i == (p.num_encoder_layers - 1)):
        params.normalize_output = p.normalize_encoder
        params.final_enc_layer = (i == (p.num_encoder_layers - 1))
      if p.packed_input:
        params.packed_input = p.packed_input
      # Use DeterministicDropoutLayer when used in temp graphs.
      if len(p.splits) > 1 or p.num_micro_batches > 1:
        params = params.cls.SetupDeterministicDropout(params)
      assert not params.has_aux_atten
      if p.is_transparent and i == 0:
        params.transparent_merger_tpl = p.transparent_merger_tpl.Copy()
      transformers.append(params)

    # Decoder layers.
    for i in range(p.num_decoder_layers):
      params = p.decoder_tpl.Copy()
      params.name = 'decoder_%d' % i
      params.mask_self_atten = True
      if p.packed_input:
        params.packed_input = p.packed_input
      if len(p.splits) > 1 or p.num_micro_batches > 1:
        params = params.cls.SetupDeterministicDropout(params)
      assert params.has_aux_atten
      transformers.append(params)
    cells = []
    cell_start = 0
    # To account for embedding layers in the pipeline.
    offset = 1
    for split, cell_end in enumerate(p.splits):
      # Layer 0 (embeddings) is always in split 0.
      sub = transformers[cell_start:(cell_end + offset)]
      if split == len(p.splits) - 1 and p.softmax_tpl:
        sub.append(p.softmax_tpl)
      cell = FeatureExtractionLayer.Params().Set(
          name='cell_{}'.format(split), sub=sub)
      cells.append(cell)
      cell_start = cell_end + offset
    p.cell_tpl = cells
    super().__init__(p)

    if p.label_smoothing:
      self.CreateChild('smoother', p.label_smoothing)

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    if self.params.label_smoothing:
      self.smoother.InstantiateVariables()
    super()._CreateChildrenVariables()

  def Logits(self, theta, inputs):
    num_splits = len(self.params.splits)
    softmax = self.children['cell_{}'.format(num_splits - 1)].softmax
    softmax_theta = theta['cell_{}'.format(num_splits - 1)].softmax
    return softmax.Logits(softmax_theta, inputs)

  def GetEncoders(self):
    encoders = []
    p = self.params
    cell_start = 0
    for split, cell_end in enumerate(p.splits):
      for encoder_id in range(cell_start, cell_end):
        if encoder_id >= p.num_encoder_layers:
          break
        encoder_l = self.children['cell_{}'.format(split)].children[
            'encoder_{}'.format(encoder_id)]
        encoders.append(encoder_l)
      cell_start = cell_end
    return encoders

  def GetDecoders(self):
    decoders = []
    p = self.params
    cell_start = 0
    for split, cell_end in enumerate(p.splits):
      for layer_id in range(cell_start, cell_end):
        decoder_id = layer_id - p.num_encoder_layers
        if decoder_id < 0:
          continue
        decoder_l = self.children['cell_{}'.format(split)].children[
            'decoder_{}'.format(decoder_id)]
        decoders.append(decoder_l)
      cell_start = cell_end
    assert len(decoders) == p.num_decoder_layers
    return decoders

  def EncoderEmbedFPropDefaultTheta(self, source_id, source_task_id=None):
    emb = self.children['cell_0'].children['emb']
    return emb.GetEncoderEmbeddingsDefaultTheta(source_id, source_task_id)

  def DecoderEmbedFPropDefaultTheta(self, tgt_id, tgt_task_id=None, t=None):
    emb = self.children['cell_0'].children['emb']
    return emb.GetDecoderEmbeddingsDefaultTheta(tgt_id, tgt_task_id, t)

  def EncoderFPropDefaultTheta(self,
                               source_vecs,
                               source_paddings,
                               source_segment_id=None,
                               source_task_id=None,
                               target_task_id=None):
    p = self.params
    transparent_acc = None
    transparent_weights = None
    for encoder_l in self.GetEncoders():
      if source_task_id is not None or target_task_id is not None:
        encoder_outs = encoder_l.FProp(
            encoder_l.theta,
            source_vecs,
            source_paddings,
            None,
            None,
            source_segment_id,
            None,
            transparent_acc,
            transparent_weights,
            source_task_id=source_task_id,
            target_task_id=target_task_id)
      else:
        encoder_outs = encoder_l.FProp(encoder_l.theta, source_vecs,
                                       source_paddings, None, None,
                                       source_segment_id, None, transparent_acc,
                                       transparent_weights)
      source_vecs = encoder_outs[0]
      if p.is_transparent and len(encoder_outs) == 8:
        transparent_acc = encoder_outs[6]
        transparent_weights = encoder_outs[7]
    return source_vecs

  def FProp(self,
            theta,
            source_input,
            source_paddings,
            target_input=None,
            target_paddings=None,
            source_segment_id=None,
            target_segment_id=None,
            labels=None,
            label_weights=None,
            source_segment_pos=None,
            target_segment_pos=None,
            source_task_id=None,
            target_task_id=None):
    """Transforms source sequence of Tensors with Transformers layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_input:  A sequence of ints indicating source input ids of [time,
        batch] shape or [batch, time] if batch_dim is 0.
      source_paddings: A sequence of 0s and 1s indicating input paddings of
        [time, batch] shape or [batch, time] if batch_dim is 0.
      target_input: A sequence of ints indicating target input ids of [time,
        batch] shape or [batch, time] if batch_dim is 0.
      target_paddings: [target_time, target_batch] or [target_batch,
        target_time] if batch_dim is 0.
      source_segment_id: A sequence of ints indicating source segment ids of
        [time, batch] shape or [batch, time] if batch_dim is 0.
      target_segment_id: A sequence of ints indicating target segment ids of
        [time, batch] shape or [batch, time] if batch_dim is 0.
      labels: A sequence of ints indicating label ids of [time, batch] shape, or
        [batch, time] if batch_dim is 0.
      label_weights: A sequence of floats indicates label weights of [time,
        batch] shape, or [batch, time] if batch_dim is 0.
      source_segment_pos: A sequence of ints indicating source position ids of
        [time, batch] shape, or [batch, time] if batch_dim is 0.
      target_segment_pos: A sequence of ints indicating target position ids of
        [time, batch] shape, or [batch, time] if batch_dim is 0.
      source_task_id: A sequence of ints indicating source task ids of [time,
        batch] shape, or [batch, time] if batch_dim is 0.
      target_task_id: A sequence of ints indicating target task ids of [time,
        batch] shape, or [batch, time] if batch_dim is 0.

    Returns:
      transformer_output with shape [time, batch, dim] or [batch, time, dim]
      if batch_dim is 0.
    """
    p = self.params
    if p.num_decoder_layers > 0:
      assert target_input is not None
      assert target_paddings is not None
    if p.packed_input:
      assert source_segment_id is not None, (
          'Need to specify src_segment_id if packed input is supported.')
      assert source_segment_pos is not None, (
          'Need to specify src_segment_pos for packed input and embeddings.')

    logits = super().FProp(theta, source_input, source_paddings, target_input,
                           target_paddings, source_segment_id,
                           target_segment_id, source_segment_pos,
                           target_segment_pos, source_task_id, target_task_id)
    if not p.softmax_tpl:
      return logits
    label_weights = tf.reshape(label_weights, [-1])
    target_probs = None
    if p.label_smoothing:
      if p.batch_dim:  # Time-major
        target_probs = tf.transpose(
            self.smoother.FProp(
                theta.smoother,
                tf.transpose(target_paddings),
                tf.transpose(labels),
                target_ids=None), [1, 0, 2])
      else:
        target_probs = self.smoother.FProp(
            theta.smoother, target_paddings, labels, target_ids=None)
      target_probs = tf.reshape(target_probs, [-1, p.softmax_tpl.num_classes])
    reshaped_logits = tf.reshape(logits, [-1, p.softmax_tpl.num_classes])
    tgt_labels = tf.reshape(labels, [-1])
    num_splits = len(p.splits)
    softmax = self.children['cell_{}'.format(num_splits - 1)].softmax
    softmax_theta = theta['cell_{}'.format(num_splits - 1)].softmax
    per_example_xent, _ = softmax.XentLossFromLogits(
        softmax_theta,
        reshaped_logits,
        class_weights=tf.reshape(label_weights, [-1]),
        class_ids=tgt_labels,
        class_probabilities=target_probs)
    xent_shape = tf.shape(logits)[:2]
    per_example_xent = tf.reshape(per_example_xent, xent_shape)
    return per_example_xent, logits


class GPipeEvolvedTransformerStack(GPipeTransformerStack):
  """Evolved Transformer stack for GPipe.

  With optional layer normalization applied to the final output.

  See 'Evolved Transformer' for more details:
  https://arxiv.org/abs/1901.11117 .
  """

  @classmethod
  def Params(cls):
    """Configs for EvolvedTransformerStack."""
    p = super().Params()
    p.encoder_tpl = GPipeEvolvedTransformerEncoderLayer.Params()
    p.decoder_tpl = GPipeEvolvedTransformerDecoderLayer.Params()
    return p


class DeterministicWeightsLayer(base_layer.BaseLayer):
  """WeightedSumLayer with deterministic dropout."""

  @classmethod
  def Params(cls):
    """Params for this MergerLayer class."""
    p = super().Params()
    p.Define('num_sources', 0, 'Number of input sources to combine.')
    p.Define('weighted_merger_dropout_prob', 0.0,
             'Applies dropout to the weights.')
    p.Define(
        'weighted_merger_softmax', True, 'If set, applies a softmax '
        'layer on top of the weights for normalization.')
    p.Define('global_weight_scale', 1.0, 'A global scale put on weights.')
    p.Define('minimal_prob', 0.0, 'The minimal weight for each component.')
    p.Define('dropout_tpl', layers.DeterministicDropoutLayer.Params(),
             'Dropout layer')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name!')

    assert p.num_sources > 0, ('Must specify num_sources > 0.')

    p.dropout_tpl.name = 'dropout'
    self.CreateChild('weighted_merger_dropout', p.dropout_tpl)

  def _CreateLayerVariables(self):
    super()._CreateLayerVariables()
    p = self.params
    params_init = py_utils.WeightInit.Constant(0.0)
    # Weights to be learned.
    pw = py_utils.WeightParams(
        shape=[p.num_sources],
        init=params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    self.CreateVariable('sum_weight', pw)

  def FProp(self, theta):
    """Combines the list of input tensors into a single tensor.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.

    Returns:
      A tensor of weights with dropout applied with shape [num_sources].
    """
    p = self.params

    # The constant factor is just meant to support the non-normalized scenario.
    # If softmax is applied, this factor will cancel out.
    w = theta.sum_weight * p.global_weight_scale + (1 / p.num_sources)
    w = tf.reshape(w, [p.num_sources])
    w = self.weighted_merger_dropout.FProp(theta.weighted_merger_dropout, w)
    if p.weighted_merger_softmax:
      residual_weights = p.minimal_prob * p.num_sources
      assert residual_weights >= 0.0
      assert residual_weights < 1.0
      w = tf.nn.softmax(w, axis=0) * (1.0 - residual_weights) + p.minimal_prob
    return w


class GPipeBatchMajorTransformerSoftmaxLayer(layers.SimpleFullSoftmax):
  """GPipe compatible softmax layer for transformers for computing logits.

  FProp interface is different for batch major stack, using segment_masks
  instead of segment_ids.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('inputs_from_decoder', False,
             'Bool, whether inputs to this layer come from decoder or not.')
    return p

  def FProp(self, theta, source_vecs, source_paddings, target_vecs,
            target_paddings, encoder_self_atten_segment_mask,
            decoder_self_atten_segment_mask, decoder_cross_atten_segment_mask):
    p = self.params
    if p.inputs_from_decoder:
      transformer_output = target_vecs
    else:
      transformer_output = source_vecs
    dim1, dim2 = tf.shape(transformer_output)[0], tf.shape(
        transformer_output)[1]
    softmax_input = tf.reshape(transformer_output, [-1, p.input_dim])
    output_shape = [dim1, dim2, p.num_classes]
    return tf.reshape(super().Logits(theta, [softmax_input]), output_shape)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    dim1, dim2 = args[1][:2] if p.inputs_from_decoder else inputs[:2]
    logits = tshape.Shape([dim1, dim2, p.num_classes])
    return py_utils.NestedMap(flops=100, out_shapes=(logits,))


class GPipeBatchMajorTransformerEmbeddingLayer(base_layer.BaseLayer):
  """GPipe compatible embeddings for transformers."""

  @classmethod
  def Params(cls):
    """Configs of Embedding layers for TransformerStack."""
    p = super().Params()
    # Note: we use the same configs for src and tgt embeddings right now.
    p.Define('token_emb',
             layers.SimpleEmbeddingLayer.Params().Set(scale_sqrt_depth=True),
             'The embedding layer params.')
    p.Define('position_emb', layers.PositionalEmbeddingLayer.Params(),
             'Position embedding layer params.')
    p.Define('input_dropout_prob', 0.0, 'Prob at which we do input dropout.')
    p.Define(
        'dropout_tpl', layers.DropoutLayer.Params(),
        'Replace with deterministic dropout for splits > 1 '
        'or microbatches > 1.')
    p.Define('add_tgt_embedding_layer', True,
             'Set True if layer embeds tgt instead of src.')
    p.Define('packed_input', True, 'Set True to support packed inputs.')
    p.Define('max_seq_len', 300, 'Max. seq len for decoding.')
    p.Define('target_vocab_size', 0, 'Target vocab size, if different.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    p.token_emb.name = 'src_token_emb'
    p.position_emb.name = 'src_position_emb'
    self.CreateChild('src_token_emb', p.token_emb)
    self.CreateChild('src_pos_emb', p.position_emb)

    p.dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
    p.dropout_tpl.name = 'src_dropout'
    self.CreateChild('src_dropout', p.dropout_tpl)

    if p.add_tgt_embedding_layer:
      params = p.token_emb.Copy()
      if p.target_vocab_size:
        params.vocab_size = p.target_vocab_size
      params.name = 'tgt_token_emb'
      self.CreateChild('tgt_token_emb', params)
      params = p.position_emb.Copy()
      params.name = 'tgt_position_emb'
      self.CreateChild('tgt_pos_emb', params)

      params = p.dropout_tpl.Copy()
      params.keep_prob = (1.0 - p.input_dropout_prob)
      params.name = 'tgt_dropout'
      self.CreateChild('tgt_dropout', params)

  def GetEmbeddings(self, emb_theta, emb, pos_emb_theta, pos_emb, dropout_theta,
                    dropout, input_ids, input_segment_pos):
    p = self.params
    time_dim = 1
    seq_len = tf.shape(input_ids)[time_dim]
    input_embs = emb.EmbLookup(emb_theta, input_ids)
    if p.packed_input:  # Packed inputs.
      pos_embs = pos_emb.FPropWithPosition(pos_emb_theta, input_segment_pos)
    else:
      pos_embs = tf.expand_dims(pos_emb.FProp(pos_emb_theta, seq_len), 0)

    input_embs += pos_embs
    input_embs = dropout.FProp(dropout_theta, input_embs)
    return input_embs

  # To be used for decoding.
  def GetEncoderEmbeddingsDefaultTheta(self, input_ids):
    seq_len = tf.shape(input_ids)[1]
    input_embs = self.src_token_emb.EmbLookup(self.theta.src_token_emb,
                                              input_ids)
    pos_embs = tf.expand_dims(
        self.src_pos_emb.FProp(self.theta.src_pos_emb, seq_len), 0)
    input_embs += pos_embs
    input_embs = self.src_dropout.FProp(self.theta.src_dropout, input_embs)
    return input_embs

  # To be used for decoding.
  def GetDecoderEmbeddingsDefaultTheta(self, input_ids, t):
    seq_len = tf.shape(input_ids)[1]
    input_embs = self.tgt_token_emb.EmbLookup(self.theta.tgt_token_emb,
                                              input_ids)
    # [target_batch, 1, dim]
    # t should be shaped as [target_batch, 1].
    if t is None:
      pos_embs = tf.expand_dims(
          self.tgt_pos_emb.FProp(self.theta.tgt_pos_emb, seq_len), 0)
    else:
      pos_embs = self.tgt_pos_emb.FPropWithPosition(self.theta.tgt_pos_emb, t)
    input_embs += pos_embs
    input_embs = self.tgt_dropout.FProp(self.theta.tgt_dropout, input_embs)
    return input_embs

  def FProp(self, theta, source_id, source_paddings, target_id, target_paddings,
            encoder_self_atten_segment_mask, decoder_self_atten_segment_mask,
            decoder_cross_atten_segment_mask, source_segment_pos,
            target_segment_pos):
    p = self.params
    with tf.name_scope(p.name):
      source_vecs = self.GetEmbeddings(theta.src_token_emb, self.src_token_emb,
                                       theta.src_pos_emb, self.src_pos_emb,
                                       theta.src_dropout, self.src_dropout,
                                       source_id, source_segment_pos)
      target_vecs = None
      if p.add_tgt_embedding_layer:
        target_vecs = self.GetEmbeddings(theta.tgt_token_emb,
                                         self.tgt_token_emb, theta.tgt_pos_emb,
                                         self.tgt_pos_emb, theta.tgt_dropout,
                                         self.tgt_dropout, target_id,
                                         target_segment_pos)
      rets = (source_vecs, source_paddings, target_vecs, target_paddings,
              encoder_self_atten_segment_mask, decoder_self_atten_segment_mask,
              decoder_cross_atten_segment_mask)
      return rets

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    # TODO(ankurbpn): return accurate estimate of flops.
    py_utils.CheckShapes((inputs,))
    flops_per_element = 2  # Is this correct?
    vocab = p.token_emb.vocab_size
    dim = p.token_emb.embedding_dim
    src_dim_0, src_dim_1 = inputs
    flops = flops_per_element * src_dim_0 * src_dim_1 * dim * vocab
    args = args if isinstance(args, tuple) else (args,)
    new_inputs = tshape.Shape([src_dim_0, src_dim_1, dim])
    new_args = list(args)
    if p.add_tgt_embedding_layer:
      tgt_dim_0, tgt_dim_1 = args[1]
      new_args[1] = tshape.Shape([tgt_dim_0, tgt_dim_1, dim])
    new_args = new_args[:6]
    new_args = tuple(new_args)
    return py_utils.NestedMap(flops=flops, out_shapes=(new_inputs,) + new_args)


class GPipeBatchMajorTransformerStack(PipeliningLayer):
  """Stacked self- multi-head attention and fully connected layers.

  With optional layer normalization applied to the final output.

  See 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
  for details.

  Implements a gipe stack for the batch major transformer variant.
  """

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super().Params()

    # GPipe Related
    p.Define(
        'splits', 1,
        'Number of splits, or list of integers specifying the ending index for '
        'each split in ascending order. Last index should be num_layers.')

    # Transformer related
    p.Define('model_dim', 1024, 'Characteristic depth (dimension).')
    p.Define('num_encoder_layers', 0, 'Number of transformer encoder layers.')
    p.Define('num_decoder_layers', 0, 'Number of transformer encoder layers.')
    p.Define('emb_tpl', GPipeBatchMajorTransformerEmbeddingLayer.Params(),
             'Prepare embeddings for Transformer input.')
    p.Define('softmax_tpl', GPipeBatchMajorTransformerSoftmaxLayer.Params(),
             'Optional softmax layer to compute the logits.')
    p.Define('label_smoothing', None, 'Label smoothing Params.')
    p.Define('encoder_tpl',
             batch_major_attention.GPipeBatchMajorTransformerLayer.Params(),
             'TransformerLayer Encoder params tpl.')
    p.Define('decoder_tpl',
             batch_major_attention.GPipeBatchMajorTransformerLayer.Params(),
             'TransformerLayer Decoder params tpl.')
    p.Define('packed_input', True,
             'If True, assumes multiple training samples per input.')
    p.encoder_tpl.has_aux_atten = False
    p.decoder_tpl.has_aux_atten = True
    p.decoder_tpl.mask_self_atten = True
    p.batch_dim = 0
    return p

  def __init__(self, params):
    p = params.Copy()
    num_layers = p.num_encoder_layers + p.num_decoder_layers

    if isinstance(p.splits, (list, tuple)):
      assert p.splits[-1] == num_layers
      for i, j in zip(p.splits[:-1], p.splits[1:]):
        assert i <= j, 'Splits must be in increasing order.'
    else:
      num_splits = p.splits
      layers_per_split = (num_layers - 1) // num_splits + 1
      p.splits = []
      for i in range(num_splits):
        p.splits.append((i + 1) * layers_per_split)
      p.splits[-1] = num_layers

    p.state_dtype = p.dtype
    if p.fprop_dtype:
      p.state_dtype = p.fprop_dtype

    transformers = []

    # Encoder Embedding layer.
    if len(p.splits) > 1 or p.num_micro_batches > 1:
      p.emb_tpl.dropout_tpl = layers.DeterministicDropoutLayer.Params()
    p.emb_tpl.packed_input = p.packed_input
    p.emb_tpl.add_tgt_embedding_layer = (p.num_decoder_layers > 0)
    p.emb_tpl.name = 'emb'
    transformers.append(p.emb_tpl)
    if p.softmax_tpl:
      p.softmax_tpl.name = 'softmax'
      p.softmax_tpl.inputs_from_decoder = p.num_decoder_layers > 0
    # Encoder layers.
    for i in range(p.num_encoder_layers):
      params = p.encoder_tpl.Copy()
      params.name = 'encoder_%d' % i
      if i == (p.num_encoder_layers - 1):
        params.output_layer_norm = True
      params.packed_input = p.packed_input
      # Use DeterministicDropoutLayer when used in temp graphs.
      if len(p.splits) > 1 or p.num_micro_batches > 1:
        params = params.cls.SetupDeterministicDropout(params)
      assert not params.has_aux_atten
      transformers.append(params)

    # Decoder layers.
    for i in range(p.num_decoder_layers):
      params = p.decoder_tpl.Copy()
      params.name = 'decoder_%d' % i
      params.mask_self_atten = True
      if i == (p.num_decoder_layers - 1):
        params.output_layer_norm = True
      params.packed_input = p.packed_input
      if len(p.splits) > 1 or p.num_micro_batches > 1:
        params = params.cls.SetupDeterministicDropout(params)
      assert params.has_aux_atten
      transformers.append(params)
    cells = []
    cell_start = 0
    # To account for embedding layers in the pipeline.
    offset = 1
    for split, cell_end in enumerate(p.splits):
      # Layer 0 (embeddings) is always in split 0.
      sub = transformers[cell_start:(cell_end + offset)]
      if split == len(p.splits) - 1 and p.softmax_tpl:
        sub.append(p.softmax_tpl)
      cell = FeatureExtractionLayer.Params().Set(
          name='cell_{}'.format(split), sub=sub)
      cells.append(cell)
      cell_start = cell_end + offset
    p.cell_tpl = cells
    super().__init__(p)

    if p.label_smoothing:
      self.CreateChild('smoother', p.label_smoothing)

  def _CreateChildrenVariables(self):
    # Backwards compatibility: manually call child.InstantiateVariables()
    # outside of tf.variable_scope(p.name).
    if self.params.label_smoothing:
      self.smoother.InstantiateVariables()
    super()._CreateChildrenVariables()

  def Logits(self, theta, inputs):
    num_splits = len(self.params.splits)
    softmax = self.children['cell_{}'.format(num_splits - 1)].softmax
    softmax_theta = theta['cell_{}'.format(num_splits - 1)].softmax
    return softmax.Logits(softmax_theta, inputs)

  def GetEncoders(self):
    encoders = []
    p = self.params
    cell_start = 0
    for split, cell_end in enumerate(p.splits):
      for encoder_id in range(cell_start, cell_end):
        if encoder_id >= p.num_encoder_layers:
          break
        encoder_l = self.children['cell_{}'.format(split)].children[
            'encoder_{}'.format(encoder_id)]
        encoders.append(encoder_l)
      cell_start = cell_end
    return encoders

  def GetDecoders(self):
    decoders = []
    p = self.params
    cell_start = 0
    for split, cell_end in enumerate(p.splits):
      for layer_id in range(cell_start, cell_end):
        decoder_id = layer_id - p.num_encoder_layers
        if decoder_id < 0:
          continue
        decoder_l = self.children['cell_{}'.format(split)].children[
            'decoder_{}'.format(decoder_id)]
        decoders.append(decoder_l)
      cell_start = cell_end
    assert len(decoders) == p.num_decoder_layers
    return decoders

  def EncoderEmbedFPropDefaultTheta(self, source_id):
    emb = self.children['cell_0'].children['emb']
    return emb.GetEncoderEmbeddingsDefaultTheta(source_id)

  def DecoderEmbedFPropDefaultTheta(self, tgt_id, t=None):
    emb = self.children['cell_0'].children['emb']
    return emb.GetDecoderEmbeddingsDefaultTheta(tgt_id, t)

  def EncoderFPropDefaultTheta(self,
                               source_vecs,
                               source_paddings,
                               encoder_self_atten_segment_mask=None):
    for encoder_l in self.GetEncoders():
      encoder_outs = encoder_l.FProp(encoder_l.theta, source_vecs,
                                     source_paddings, None, None,
                                     encoder_self_atten_segment_mask, None,
                                     None)
      source_vecs = encoder_outs[0]
    return source_vecs

  def FProp(self,
            theta,
            source_input,
            source_paddings,
            target_input=None,
            target_paddings=None,
            source_segment_id=None,
            target_segment_id=None,
            labels=None,
            label_weights=None,
            source_segment_pos=None,
            target_segment_pos=None):
    """Transforms source sequence of Tensors with Transformers layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_input:  A sequence of ints indicating source input ids of [batch,
        time].
      source_paddings: A sequence of 0s and 1s indicating input paddings of
        [batch, time].
      target_input: A sequence of ints indicating target input ids of [batch,
        time].
      target_paddings: [target_batch, target_time].
      source_segment_id: A sequence of ints indicating source segment ids of
        [batch, time].
      target_segment_id: A sequence of ints indicating target segment ids of
        [batch, time].
      labels: A sequence of ints indicating label ids of [batch, time].
      label_weights: A sequence of floats indicates label weights of [batch,
        time].
      source_segment_pos: A sequence of ints indicating source position ids of
        [batch, time].
      target_segment_pos: A sequence of ints indicating target position ids of
        [batch, time].

    Returns:
      transformer_output with shape [batch, time, dim].
    """
    p = self.params
    if p.num_decoder_layers > 0:
      assert target_input is not None
      assert target_paddings is not None
      target_time = tf.shape(target_input)[1]
      batch = tf.shape(target_input)[0]
    encoder_self_atten_segment_mask = None
    decoder_self_atten_segment_mask = None
    decoder_cross_atten_segment_mask = None

    # Prepare segment masks from segment ids.
    if p.packed_input:
      dtype = py_utils.FPropDtype(p)
      assert source_segment_id is not None, (
          'Need to specify src_segment_id if packed input is supported.')
      assert source_segment_pos is not None, (
          'Need to specify src_segment_pos for packed input and embeddings.')
      encoder_self_atten_segment_mask = batch_major_attention.SegmentMask(
          source_segment_id, source_segment_id, dtype, False)
      if target_segment_id is not None:
        decoder_self_atten_segment_mask = batch_major_attention.SegmentMask(
            target_segment_id, target_segment_id, dtype, False)
        causal_padding = tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    batch_major_attention.CausalPadding(
                        target_time, dtype=dtype), 0), [batch, 1, 1]), 1)
        decoder_self_atten_segment_mask = tf.math.maximum(
            causal_padding, decoder_self_atten_segment_mask)
        decoder_cross_atten_segment_mask = batch_major_attention.SegmentMask(
            target_segment_id, source_segment_id, dtype, False)

    # FProp through the gpipe pipeline.
    logits = super().FProp(theta, source_input, source_paddings, target_input,
                           target_paddings, encoder_self_atten_segment_mask,
                           decoder_self_atten_segment_mask,
                           decoder_cross_atten_segment_mask, source_segment_pos,
                           target_segment_pos)
    label_weights = tf.reshape(label_weights, [-1])
    target_probs = None
    if p.label_smoothing:
      target_probs = self.smoother.FProp(
          theta.smoother, target_paddings, labels, target_ids=None)
      target_probs = tf.reshape(target_probs, [-1, p.softmax_tpl.num_classes])
    reshaped_logits = tf.reshape(logits, [-1, p.softmax_tpl.num_classes])
    tgt_labels = tf.reshape(labels, [-1])
    num_splits = len(p.splits)
    softmax = self.children['cell_{}'.format(num_splits - 1)].softmax
    softmax_theta = theta['cell_{}'.format(num_splits - 1)].softmax
    per_example_xent, _ = softmax.XentLossFromLogits(
        softmax_theta,
        reshaped_logits,
        class_weights=tf.reshape(label_weights, [-1]),
        class_ids=tgt_labels,
        class_probabilities=target_probs)
    xent_shape = tf.shape(logits)[:2]
    per_example_xent = tf.reshape(per_example_xent, xent_shape)
    return per_example_xent, logits
