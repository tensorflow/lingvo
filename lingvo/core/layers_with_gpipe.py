# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import layers_with_attention
from lingvo.core import py_utils
from lingvo.core import tshape
from lingvo.core.gpipe import FeatureExtractionLayer
from lingvo.core.gpipe import PipeliningLayer
from six.moves import range
from six.moves import zip
import tensorflow as tf


def _common_gpipe_transformer_params(p):
  """Add GPipe params to layer."""
  p.Define(
      'is_transparent', False,
      'If set, encoder outputs a list of layer outputs while decoder '
      'expects a list of source input vectors.')
  p.Define(
      'num_transparent_outputs', 0,
      'Number of transparent outputs. Only positive if this is the '
      'last encoder')
  p.Define('transparent_merger_tpl', None,
           'Merger op for layer outputs. Not none if this is the last encoder')
  return p


def _common_gpipe_transformer_init(layer):
  """Initialize a GPipe layer."""
  p = layer.params
  if p.is_transparent and p.num_transparent_outputs > 0:
    transparent_params = []
    for i in range(p.num_transparent_outputs):
      transparent_param = p.transparent_merger_tpl.Copy()
      transparent_param.name = 'transparent_%d' % i
      transparent_params.append(transparent_param)
    layer.CreateChildren('transparent_merger', transparent_params)
  assert p.name


def _common_gpipe_transformer_encoder_fprop(layer, layer_class, theta,
                                            source_vecs, source_paddings,
                                            target_vecs, target_paddings,
                                            source_segment_id,
                                            target_segment_id, labels,
                                            label_weights, *more_source_vecs):
  """GPipe encoder FProp."""
  p = layer.params
  h, _ = super(layer_class, layer).FProp(
      theta, source_vecs, source_paddings, source_segment_id=source_segment_id)
  h.set_shape(source_vecs.shape)
  if p.is_transparent:
    more_source_vecs += (source_vecs,)
    if p.num_transparent_outputs > 0:  # Merger layer.
      transformer_output = []
      for i in range(p.num_transparent_outputs):
        merged_outputs = layer.transparent_merger[i].FProp(
            theta.transparent_merger[i], list(more_source_vecs + (h,)))
        transformer_output.append(merged_outputs)
      h = transformer_output[0]
      if p.num_transparent_outputs == 1:
        more_source_vecs = ()
      else:
        more_source_vecs = tuple(transformer_output[1:])
  return (h, source_paddings, target_vecs, target_paddings, source_segment_id,
          target_segment_id, labels, label_weights) + more_source_vecs


def _common_gpipe_transformer_decoder_fprop(layer, layer_class, params, theta,
                                            source_vecs, source_paddings,
                                            target_vecs, target_paddings,
                                            source_segment_id,
                                            target_segment_id, labels,
                                            label_weights, *more_source_vecs):
  """GPipe decoder FProp."""
  assert target_vecs is not None
  assert target_paddings is not None
  h, _ = super(layer_class, layer).FProp(
      theta,
      target_vecs,
      target_paddings,
      aux_vecs=source_vecs,
      aux_paddings=source_paddings,
      source_segment_id=target_segment_id,
      aux_segment_id=source_segment_id)
  h.set_shape(target_vecs.shape)
  if params.is_transparent and more_source_vecs:
    source_vecs = more_source_vecs[0]
    more_source_vecs = more_source_vecs[1:]
  return (source_vecs, source_paddings, h, target_paddings, source_segment_id,
          target_segment_id, labels, label_weights) + more_source_vecs


def _common_gpipe_transformer_fprop_meta(p, inputs, *args):
  """GPipe FPropMeta function."""
  # TODO(huangyp): return accurate estimate of flops.
  py_utils.CheckShapes((inputs,))
  flops_per_element = 5
  src_time, source_batch, dim = inputs
  flops = flops_per_element * src_time * src_time * source_batch * dim
  args = args if isinstance(args, tuple) else (args,)
  if p.is_transparent:
    if p.has_aux_atten:  # Decoder FPropMeta
      args = args[:-1] if len(args) > 7 else args
    else:
      if p.num_transparent_outputs == 0:
        args += (inputs,)
      elif p.num_transparent_outputs == 1:
        # Switch back to non-transparent mode for decoder.
        args = args[:7]
      else:
        args += (inputs,) * (p.num_transparent_outputs - len(args) + 6)
  return py_utils.NestedMap(flops=flops, out_shapes=(inputs,) + args)


class GPipeTransformerLayer(layers_with_attention.TransformerLayer):
  """GPipe compatible transformer layer."""

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super(GPipeTransformerLayer, cls).Params()
    return _common_gpipe_transformer_params(p)

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerLayer, self).__init__(params)
    _common_gpipe_transformer_init(self)

  def FProp(self, theta, source_vecs, source_paddings, target_vecs,
            target_paddings, source_segment_id, target_segment_id, labels,
            label_weights, *more_source_vecs):
    p = self.params
    with tf.name_scope(p.name):
      if p.has_aux_atten:  # Decoder FProp
        return _common_gpipe_transformer_decoder_fprop(
            self, GPipeTransformerLayer, p, theta, source_vecs, source_paddings,
            target_vecs, target_paddings, source_segment_id, target_segment_id,
            labels, label_weights, *more_source_vecs)
      else:  # Encoder FProp
        return _common_gpipe_transformer_encoder_fprop(
            self, GPipeTransformerLayer, theta, source_vecs, source_paddings,
            target_vecs, target_paddings, source_segment_id, target_segment_id,
            labels, label_weights, *more_source_vecs)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    return _common_gpipe_transformer_fprop_meta(p, inputs, *args)


class GPipeEvolvedTransformerEncoderLayer(
    layers_with_attention.EvolvedTransformerEncoderLayer):
  """GPipe-compatible Evolved Transformer encoder layer."""

  @classmethod
  def Params(cls):
    p = super(GPipeEvolvedTransformerEncoderLayer, cls).Params()
    return _common_gpipe_transformer_params(p)

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeEvolvedTransformerEncoderLayer, self).__init__(params)
    _common_gpipe_transformer_init(self)

  def FProp(self, theta, source_vecs, source_paddings, source_segment_id,
            labels, label_weights, *more_source_vecs):
    with tf.name_scope(self.params.name):
      return _common_gpipe_transformer_encoder_fprop(
          self, GPipeEvolvedTransformerEncoderLayer, theta, source_vecs,
          source_paddings, None, None, source_segment_id, None, labels,
          label_weights, *more_source_vecs)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    return _common_gpipe_transformer_fprop_meta(p, inputs, *args)


class GPipeEvolvedTransformerDecoderLayer(
    layers_with_attention.EvolvedTransformerDecoderLayer):
  """GPipe-compatible Evolved Transformer decoder layer."""

  @classmethod
  def Params(cls):
    p = super(GPipeEvolvedTransformerDecoderLayer, cls).Params()
    return _common_gpipe_transformer_params(p)

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeEvolvedTransformerDecoderLayer, self).__init__(params)
    _common_gpipe_transformer_init(self)

  def FProp(self, theta, source_vecs, source_paddings, target_vecs,
            target_paddings, source_segment_id, target_segment_id, labels,
            label_weights, *more_source_vecs):
    with tf.name_scope(self.params.name):
      return _common_gpipe_transformer_decoder_fprop(
          self, GPipeEvolvedTransformerDecoderLayer, self.params, theta,
          source_vecs, source_paddings, target_vecs, target_paddings,
          source_segment_id, target_segment_id, labels, label_weights,
          *more_source_vecs)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    return _common_gpipe_transformer_fprop_meta(p, inputs, *args)


class GPipeTransformerSoftmaxLayer(base_layer.BaseLayer):
  """GPipe compatible softmax layer for transformers."""

  @classmethod
  def Params(cls):
    p = super(GPipeTransformerSoftmaxLayer, cls).Params()
    p.Define('num_shards', 16,
             'num_shards for softmax. Assert vocab_size % num_shards == 0')
    p.Define('label_smoothing', None, 'Label smoothing Params.')
    p.Define('softmax', layers.SimpleFullSoftmax.Params(),
             'Params for the softmax layer.')
    p.Define('inputs_from_decoder', False,
             'Bool, whether inputs to this layer come from decoder or not.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerSoftmaxLayer, self).__init__(params)
    p = self.params
    self.CreateChild('softmax', p.softmax)
    if p.label_smoothing is not None:
      self.CreateChild('smoother', p.label_smoothing)

  def FProp(self, theta, source_vecs, source_paddings, target_vecs,
            target_paddings, source_segment_id, target_segment_id, labels,
            label_weights, *more_source_vecs):
    p = self.params
    if p.inputs_from_decoder:
      transformer_output = target_vecs
    else:
      transformer_output = source_vecs
      if more_source_vecs:
        transformer_output = more_source_vecs + (source_vecs,)
    return self._FPropSoftmax(theta, transformer_output, labels, label_weights,
                              target_paddings)

  def _FPropSoftmax(self, theta, softmax_input, target_labels, target_weights,
                    target_paddings):
    """Computes cross-entropy loss given the softmax input, labels and weights.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      softmax_input: A tensor of shape [time, batch, p.softmax.input_dim].
      target_labels: A matrix of params.dtype [time, batch].
      target_weights: A matrix of params.dtype [time, batch].
      target_paddings: A matrix of params.dtype [time, batch].

    Returns:
      per_example_xent: Tensor with shape [time, batch]
      logits: Tensor with shape [time, batch, vocab_size]
    """
    p = self.params

    if p.label_smoothing is None:
      xent_loss = self.softmax.FProp(
          theta.softmax, [softmax_input],
          class_weights=target_weights,
          class_ids=tf.cast(target_labels, tf.int32))
    else:
      # [time, batch, num_classes]
      target_probs = tf.transpose(
          self.smoother.FProp(
              theta.smoother,
              tf.transpose(target_paddings),
              tf.transpose(target_labels),
              target_ids=None), [1, 0, 2])
      xent_loss = self.softmax.FProp(
          theta.softmax, [softmax_input],
          class_weights=target_weights,
          class_probabilities=target_probs)
    return xent_loss.per_example_xent, xent_loss.logits

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    t, b = args[1][:2] if p.inputs_from_decoder else inputs[:2]
    per_example_xent = tshape.Shape([t, b])
    logits = tshape.Shape([t, b, p.softmax.num_classes])
    return py_utils.NestedMap(flops=100, out_shapes=(per_example_xent, logits))


class GPipeTransformerEmbeddingLayer(base_layer.BaseLayer):
  """GPipe compatible embeddings for transformers."""

  @classmethod
  def Params(cls):
    """Configs of Embedding layers for TransformerStack."""
    p = super(GPipeTransformerEmbeddingLayer, cls).Params()
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
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerEmbeddingLayer, self).__init__(params)
    p = self.params
    with tf.variable_scope(p.name):
      p.token_emb.name = 'src_token_emb'
      p.position_emb.name = 'src_position_emb'
      self.CreateChild('src_token_emb', p.token_emb)
      self.CreateChild('src_pos_emb', p.position_emb)

      p.dropout_tpl.keep_prob = (1.0 - p.input_dropout_prob)
      p.dropout_tpl.name = 'src_dropout'
      self.CreateChild('src_dropout', p.dropout_tpl)

      if p.add_tgt_embedding_layer:
        params = p.token_emb.Copy()
        params.name = 'tgt_token_emb'
        self.CreateChild('tgt_token_emb', params)
        params = p.position_emb.Copy()
        params.name = 'tgt_position_emb'
        self.CreateChild('tgt_pos_emb', params)

        params = p.dropout_tpl.Copy()
        params.keep_prob = (1.0 - p.input_dropout_prob)
        params.name = 'tgt_dropout'
        self.CreateChild('tgt_dropout', params)
    assert p.name

  def GetEmbeddings(self, emb_theta, emb, pos_emb_theta, pos_emb, dropout_theta,
                    dropout, input_ids, input_pos_ids):
    p = self.params
    seq_len = tf.shape(input_ids)[0]
    # [seq_len, batch, model_dim]
    input_embs = emb.EmbLookup(emb_theta, input_ids)
    if p.packed_input:  # Packed inputs.
      # [seq_len, batch, dim] or [batch, dim] in case of beam search.
      pos_embs = pos_emb.FPropWithPosition(pos_emb_theta, input_pos_ids)
    else:
      # [seq_len, 1, model_dim]
      pos_embs = tf.expand_dims(pos_emb.FProp(pos_emb_theta, seq_len), 1)

    input_embs += pos_embs
    input_embs = dropout.FProp(dropout_theta, input_embs)
    return input_embs

  # To be used for decoding.
  def GetEncoderEmbeddingsDefaultTheta(self, input_ids):
    seq_len = tf.shape(input_ids)[0]
    # [seq_len, batch, model_dim]
    input_embs = self.src_token_emb.EmbLookup(self.theta.src_token_emb,
                                              input_ids)
    # [seq_len, 1, model_dim]
    pos_embs = tf.expand_dims(
        self.src_pos_emb.FProp(self.theta.src_pos_emb, seq_len), 1)
    input_embs += pos_embs
    input_embs = self.src_dropout.FProp(self.theta.src_dropout, input_embs)
    return input_embs

  # To be used for decoding.
  def GetDecoderEmbeddingsDefaultTheta(self, input_ids, t=None):
    p = self.params
    seq_len = tf.shape(input_ids)[0]
    # [seq_len, batch, model_dim]
    input_embs = self.tgt_token_emb.EmbLookup(self.theta.tgt_token_emb,
                                              input_ids)
    # [seq_len, 1, model_dim]
    if t is None:
      pos_embs = tf.expand_dims(
          self.tgt_pos_emb.FProp(self.theta.tgt_pos_emb, seq_len), 1)
    else:  # Support decoding.
      pos_embs = tf.slice(
          self.tgt_pos_emb.FProp(self.theta.tgt_pos_emb, p.max_seq_len), [t, 0],
          [1, p.token_emb.embedding_dim])
    input_embs += pos_embs
    input_embs = self.tgt_dropout.FProp(self.theta.tgt_dropout, input_embs)
    return input_embs

  def FProp(self, theta, source_id, source_paddings, target_id, target_paddings,
            source_segment_id, target_segment_id, labels, label_weights,
            source_pos_id, target_pos_id):
    p = self.params
    with tf.name_scope(p.name):
      source_vecs = self.GetEmbeddings(theta.src_token_emb, self.src_token_emb,
                                       theta.src_pos_emb, self.src_pos_emb,
                                       theta.src_dropout, self.src_dropout,
                                       source_id, source_pos_id)
      target_vecs = None
      if p.add_tgt_embedding_layer:
        target_vecs = self.GetEmbeddings(theta.tgt_token_emb,
                                         self.tgt_token_emb, theta.tgt_pos_emb,
                                         self.tgt_pos_emb, theta.tgt_dropout,
                                         self.tgt_dropout, target_id,
                                         target_pos_id)
      return (source_vecs, source_paddings, target_vecs, target_paddings,
              source_segment_id, target_segment_id, labels, label_weights)

  @classmethod
  def FPropMeta(cls, p, inputs, *args):
    # TODO(ankurbpn): return accurate estimate of flops.
    py_utils.CheckShapes((inputs,))
    flops_per_element = 2  # Is this correct?
    vocab = p.token_emb.vocab_size
    dim = p.token_emb.embedding_dim
    src_time, source_batch = inputs
    flops = flops_per_element * src_time * source_batch * dim * vocab
    args = args if isinstance(args, tuple) else (args,)
    new_inputs = tshape.Shape([src_time, source_batch, dim])
    new_args = list(args)
    if p.add_tgt_embedding_layer:
      tgt_time, tgt_batch = args[1]
      new_args[1] = tshape.Shape([tgt_time, tgt_batch, dim])
    new_args = tuple(new_args[:7])
    return py_utils.NestedMap(flops=flops, out_shapes=(new_inputs,) + new_args)


class GPipeTransformerStack(PipeliningLayer):
  """Stacked self- multi-head attention and fully connected layers.

  With optional layer normalization applied to the final output.

  See 'Attention Is All You Need' https://arxiv.org/abs/1706.03762
  for details.
  """

  @classmethod
  def Params(cls):
    """Configs for TransformerStack."""
    p = super(GPipeTransformerStack, cls).Params()

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
    p.Define(
        'num_transparent_outputs', 0,
        'If set, the transparent merger outputs this number of weighted sums. '
        'Defaults to number of decoder layers if transparent.')
    p.Define('packed_input', False,
             'If True, assumes multiple training samples per input.')
    p.Define('apply_dropout_every_n', 1, 'Deprecated')
    p.encoder_tpl.has_aux_atten = False
    p.decoder_tpl.has_aux_atten = True
    p.decoder_tpl.mask_self_atten = True
    p.batch_dim = 1
    return p

  @base_layer.initializer
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

    with tf.variable_scope(p.name):
      p.encoder_tpl.source_dim = p.model_dim
      p.decoder_tpl.source_dim = p.model_dim
      transformers = []

      # Encoder Embedding layer.
      if len(p.splits) > 1 or p.num_micro_batches > 1:
        p.emb_tpl.dropout_tpl = layers.DeterministicDropoutLayer.Params()
      p.emb_tpl.packed_input = p.packed_input
      p.emb_tpl.is_transparent = p.is_transparent
      p.emb_tpl.add_tgt_embedding_layer = (p.num_decoder_layers > 0)
      p.emb_tpl.name = 'emb'
      transformers.append(p.emb_tpl)
      if p.softmax_tpl:
        p.softmax_tpl.name = 'softmax'
        p.softmax_tpl.inputs_from_decoder = p.num_decoder_layers > 0
      # Encoder layers.
      for i in range(p.num_encoder_layers):
        params = p.encoder_tpl.Copy()
        params.name = 'encoder_%d' % (i)
        params.is_transparent = p.is_transparent
        params.packed_input = p.packed_input
        # Use DeterministicDropoutLayer when used in temp graphs.
        if len(p.splits) > 1 or p.num_micro_batches > 1:
          params = self.SetupDeterministicDropout(params)
        assert not params.has_aux_atten
        last_layer = (i == p.num_encoder_layers - 1)
        if p.is_transparent and last_layer:
          transparent_merger_tpl = DeterministicWeightedSumLayer.Params()
          transparent_merger_tpl.num_sources = p.num_encoder_layers + 1
          transparent_merger_tpl.dropout_tpl.keep_prob = (
              1 - p.transparent_merger_dropout_prob)
          params.transparent_merger_tpl = transparent_merger_tpl
          params.num_transparent_outputs = p.num_transparent_outputs
        transformers.append(params)

      # Decoder layers.
      for i in range(p.num_decoder_layers):
        params = p.decoder_tpl.Copy()
        params.name = 'decoder_%d' % (i)
        params.mask_self_atten = True
        params.packed_input = p.packed_input
        params.is_transparent = p.is_transparent and (
            p.num_transparent_outputs == p.num_decoder_layers)
        if len(p.splits) > 1 or p.num_micro_batches > 1:
          params = self.SetupDeterministicDropout(params)
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
    super(GPipeTransformerStack, self).__init__(p)

  def SetupDeterministicDropout(self, params):
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

  def Logits(self, inputs):
    num_splits = len(self.params.splits)
    child = self.children['cell_{}'.format(num_splits - 1)].softmax.softmax
    return child.Logits(child.theta, inputs)

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
                               source_segment_id=None):
    p = self.params
    more_source_vecs = ()
    for encoder_l in self.GetEncoders():
      encoder_outs = encoder_l.FProp(encoder_l.theta, source_vecs,
                                     source_paddings, None, None, None, None,
                                     None, None, *more_source_vecs)
      source_vecs = encoder_outs[0]
      more_source_vecs = encoder_outs[8:]

    assert p.is_transparent or not more_source_vecs

    if p.is_transparent and p.num_transparent_outputs > 1:
      source_vecs = more_source_vecs + (source_vecs,)
      if p.is_eval:
        source_vecs = tf.stack(list(source_vecs), 3)
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
            source_pos_id=None,
            target_pos_id=None):
    """Transforms source sequence of Tensors with Transformers layers.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      source_input:  A sequence of ints indicating source input ids of
        [time, batch] shape.
      source_paddings: A sequence of 0s and 1s indicating input paddings of
        [time, batch] shape.
      target_input: A sequence of ints indicating target input ids of [time,
        batch] shape.
      target_paddings: [target_time, target_batch]
      source_segment_id: A sequence of ints indicating source segment ids of
        [time, batch] shape.
      target_segment_id: A sequence of ints indicating target segment ids of
        [time, batch] shape.
      labels: A sequence of ints indicating label ids of [time, batch] shape.
      label_weights: A sequence of floats indicates label weights of
        [time, batch] shape.
      source_pos_id: A sequence of ints indicating source position ids of [time,
        batch] shape.
      target_pos_id: A sequence of ints indicating target position ids of [time,
        batch] shape.

    Returns:
      transformer_output with shape [time, batch, dim]
    """
    p = self.params
    if p.num_decoder_layers > 0:
      assert target_input is not None
      assert target_paddings is not None
    if p.packed_input:
      assert source_segment_id is not None, (
          'Need to specify src_segment_id if packed input is supported.')

    gpipe_outputs = super(GPipeTransformerStack,
                          self).FProp(theta, source_input, source_paddings,
                                      target_input, target_paddings,
                                      source_segment_id, target_segment_id,
                                      labels, label_weights, source_pos_id,
                                      target_pos_id)
    return gpipe_outputs


class DeterministicWeightedSumLayer(base_layer.BaseLayer):
  """WeightedSumLayer with deterministic dropout."""

  @classmethod
  def Params(cls):
    """Params for this MergerLayer class."""
    p = super(DeterministicWeightedSumLayer, cls).Params()
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

  @base_layer.initializer
  def __init__(self, params):
    super(DeterministicWeightedSumLayer, self).__init__(params)
    p = self.params
    if not p.name:
      raise ValueError('Layer must have a specified name!')

    assert p.num_sources > 0, ('Must specify num_sources > 0.')
    params_init = py_utils.WeightInit.Constant(0.0)
    # Weights to be learned.
    pw = py_utils.WeightParams(
        shape=[p.num_sources],
        init=params_init,
        dtype=p.dtype,
        collections=[self.__class__.__name__ + '_vars'])
    with tf.variable_scope(p.name):
      self.CreateVariable('sum_weight', pw)
    p.dropout_tpl.name = 'dropout'
    self.CreateChild('weighted_merger_dropout', p.dropout_tpl)

  def FProp(self, theta, inputs):
    """Combines the list of input tensors into a single tensor.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      inputs: A list of tensors of shape [time, batch, hidden_dim]

    Returns:
      A tensor of the same shape with input tensors.
    """
    p = self.params
    n_sources = len(inputs)

    if n_sources == 1:
      return inputs[0]

    # Weighted sum of all sources, all dims must match.
    # For weighted_sum, assume input is a list of rank 3 tensors
    inputs = tf.stack(inputs)
    inputs = py_utils.HasRank(inputs, 4)

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

    w = tf.reshape(w, [p.num_sources, 1, 1, 1])
    output = tf.reduce_sum(inputs * w, axis=0)

    return output
